import uuid
from datetime import datetime
from uuid import UUID

from pydantic import BaseModel

from server.models import ChatMessage, ChatRole, DrugResponseDTO

RETRY_TIMEOUT = 15000


def enumerate_masks(sentence):
    return sentence.count("MASK")


def sse_format(obj: BaseModel) -> dict[str, str | int | UUID]:
    return {
        "event": "new_message",
        "retry": RETRY_TIMEOUT,
        "data": obj.model_dump_json(),
    }


def make_log(uid: str, text: str) -> ChatMessage:
    return ChatMessage(
        id=uid,
        role=ChatRole.assistant,
        text=text,
        createdAt=datetime.now(),
        pending=True,
        error=False,
    )


def make_error(uid: str, text: str) -> ChatMessage:
    return ChatMessage(
        id=uid,
        role=ChatRole.assistant,
        text=text,
        createdAt=datetime.now(),
        pending=False,
        error=True,
    )


def make_final(uid: str, dr: DrugResponseDTO, text: str) -> ChatMessage:
    return ChatMessage(
        id=uid,
        role=ChatRole.assistant,
        text=text,
        createdAt=datetime.now(),
        pending=False,
        drugResponse=dr,
    )


def get_uuid() -> str:
    return str(uuid.uuid4())
