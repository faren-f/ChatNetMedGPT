FROM pytorch/pytorch:latest

WORKDIR /app


COPY requirements.txt /app/

COPY .env /app/


RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

COPY src /app/src
COPY server_main.py /app/
COPY main /app/main

CMD ["uvicorn", "server_main:app", "--reload","--host", "0.0.0.0", "--port", "8000", "--proxy-headers","--root-path", "chatnetmedgpt-api", "--forwarded-allow-ips='*'"]
