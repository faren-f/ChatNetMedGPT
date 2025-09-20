# app.py
import asyncio
import csv
import shlex
import subprocess
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse

APP_TITLE = "ChatNetMedGPT API"
APP_DESCRIPTION = """
**ChatNetMedGPT runner**

To run the underlying model directly:The output CSV is written to `data/user_response/`.
"""

app = FastAPI(title=APP_TITLE, description=APP_DESCRIPTION)

MODEL_SCRIPT = Path("ChatNetMedGPT/netmedgpt_llm.py")
OUTPUT_DIR = Path("data/user_response")


def latest_csv(dirpath: Path) -> Optional[Path]:
    files = sorted(dirpath.glob("*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
    return files[0] if files else None


def preview_csv(path: Path, limit: int = 5) -> List[dict]:
    rows: List[dict] = []
    try:
        with path.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader):
                if i >= limit:
                    break
                rows.append(row)
    except Exception:
        # If preview fails, just return empty
        pass
    return rows


def run_model(user_text: str, timeout: int = 600) -> subprocess.CompletedProcess:
    if not MODEL_SCRIPT.exists():
        raise FileNotFoundError(
            f"Model script not found at {MODEL_SCRIPT.resolve()}. "
            "Ensure you run from repo root or fix MODEL_SCRIPT path."
        )
    cmd = f'python {shlex.quote(str(MODEL_SCRIPT))} --user_text {shlex.quote(user_text)}'
    return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)


@app.get(
    "/run",
    summary="Run ChatNetMedGPT once and return CSV info",
    description="Invokes the model script with the provided `user_text` and returns the path to the generated CSV plus a small preview.",
)
def run_once(
        user_text: str = Query(..., min_length=1, description="User query passed to the model."),
        preview_rows: int = Query(5, ge=0, le=50,
                                  description="Number of preview rows to include from the CSV."),
):
    try:
        proc = run_model(user_text)
    except FileNotFoundError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except subprocess.TimeoutExpired:
        raise HTTPException(status_code=504, detail="Model execution timed out.")

    if proc.returncode != 0:
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": "Model script failed",
                "stderr": proc.stderr[-5000:],  # last 5k chars to avoid flooding
            },
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = latest_csv(OUTPUT_DIR)
    if not csv_path:
        raise HTTPException(status_code=500, detail="No CSV produced in data/user_response/")

    return {
        "status": "ok",
        "user_text": user_text,
        "csv_path": str(csv_path),
        "preview": preview_csv(csv_path, limit=preview_rows) if preview_rows else [],
        "stderr_tail": proc.stderr[-1000:],  # often helpful for debugging warnings
    }


# --- WebSocket chat that runs the model per message and returns CSV info ---
@app.websocket("/ws")
async def ws_chat(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            user_text = await websocket.receive_text()
            # Run the model in a thread (avoid blocking the event loop)
            try:
                proc = await asyncio.to_thread(run_model, user_text)
            except FileNotFoundError as e:
                await websocket.send_json({"type": "error", "message": str(e)})
                continue
            except subprocess.TimeoutExpired:
                await websocket.send_json(
                    {"type": "error", "message": "Model execution timed out."})
                continue

            if proc.returncode != 0:
                await websocket.send_json(
                    {
                        "type": "error",
                        "message": "Model script failed",
                        "stderr_tail": proc.stderr[-2000:],
                    }
                )
                continue

            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            csv_path = latest_csv(OUTPUT_DIR)
            if not csv_path:
                await websocket.send_json(
                    {"type": "error", "message": "No CSV produced in data/user_response/."}
                )
                continue

            await websocket.send_json(
                {
                    "type": "result",
                    "user_text": user_text,
                    "csv_path": str(csv_path),
                    "preview": preview_csv(csv_path, limit=3),
                }
            )
    except WebSocketDisconnect:
        # client closed
        pass
