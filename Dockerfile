FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc libgomp1 libffi-dev && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN python -m pip install --no-cache-dir --upgrade pip && \
    python -m pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PYTHONPATH=/app
ENV CS_LOG_DIR=/tmp/cs-logs
ENV MPLCONFIGDIR=/tmp/matplotlib

RUN mkdir -p /tmp/cs-logs /tmp/matplotlib data models results

CMD ["python", "tests/smoke_openclaw_contract.py"]
