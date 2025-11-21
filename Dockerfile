FROM pytorch/pytorch:latest

WORKDIR /app


COPY main /app/main
COPY requirements.txt /app/
COPY server_main.py /app/
COPY .env /app/
COPY src /app/src

RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt


CMD ["uvicorn", "server_main:app", "--reload","--host", "0.0.0.0", "--port", "8000"]
