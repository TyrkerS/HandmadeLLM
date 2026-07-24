# Serving image for the HandmadeLLM FastAPI endpoint (CPU by default — portable).
# Build:  docker build -t handmadellm .
# Run:    docker run -p 8000:8000 -v $PWD/checkpoints:/app/checkpoints \
#              -e HLLM_CKPT=checkpoints/tinystories_30m/best.pt handmadellm
FROM python:3.11-slim

WORKDIR /app

# CPU torch keeps the image small and host-agnostic; swap for the cu128 wheel
# to serve on a GPU host.
RUN pip install --no-cache-dir torch==2.11.0 --index-url https://download.pytorch.org/whl/cpu
COPY requirements.txt requirements-serve.txt ./
RUN pip install --no-cache-dir numpy PyYAML \
    && pip install --no-cache-dir -r requirements-serve.txt

COPY llm/ ./llm/
COPY serve/ ./serve/

ENV HLLM_CKPT=checkpoints/tinystories_30m/best.pt
EXPOSE 8000
CMD ["uvicorn", "serve.app:app", "--host", "0.0.0.0", "--port", "8000"]
