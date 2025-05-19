FROM python:3.11-slim

# 시스템 패키지 설치
RUN apt update && apt install -y \
    build-essential \
    gcc \
    libpq-dev \
    libffi-dev \
    libssl-dev \
    rustc \
    && apt clean

WORKDIR /app

COPY requirements.txt .
COPY entrypoint.sh .
COPY .env .

RUN chmod +x entrypoint.sh
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

COPY app .

EXPOSE 8000

CMD ["./entrypoint.sh"]
