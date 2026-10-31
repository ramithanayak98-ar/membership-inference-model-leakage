FROM python:3.12-slim

LABEL maintainer="Ramitha Nayak MV (PES1UG24AM392), Shreya N Hegde (PES1UG24AM401)"
LABEL description="Reproducible Pipeline for Membership Inference and Model Leakage Evaluation"

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt || true

COPY . .

CMD ["python3", "run_pipeline.py"]
