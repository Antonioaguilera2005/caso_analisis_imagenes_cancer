FROM python:3.11-slim
WORKDIR /app
COPY requirements-web.txt .
RUN pip install --no-cache-dir -r requirements-web.txt
COPY modelo.py ./
COPY webapp ./webapp
ENV MODEL_DIR=/app/artifacts/dashboard PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2
CMD ["sh", "-c", "uvicorn webapp.main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1 --no-access-log"]
