FROM python:3.11-slim

# libgomp1: LightGBM cần OpenMP runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 curl && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ /app/src/
ENV PYTHONPATH=/app
EXPOSE 7860
CMD ["python", "-u", "/app/src/app.py"]
