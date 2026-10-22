FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
COPY churn_src/  churn_src/
COPY artifacts/  artifacts/

RUN pip install  --no-cache-dir -r requirements.txt
