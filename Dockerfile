# Image pour le service de serving (API FastAPI)
FROM python:3.11-slim

WORKDIR /app

# Dépendances système minimales pour psycopg2 / xgboost
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential libpq-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY models/ ./models/

ENV PYTHONUNBUFFERED=1
ENV DATABASE_URL=postgresql+psycopg2://fraud_user:fraud_pass@postgres:5432/fraud_db

EXPOSE 8000

CMD ["uvicorn", "src.serving.api:app", "--host", "0.0.0.0", "--port", "8000"]
