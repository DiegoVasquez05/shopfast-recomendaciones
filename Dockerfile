# TelecomBot AI — imagen del backend conversacional (también usada por el Siebel worker y el simulador de Siebel)
FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
WORKDIR /app
COPY requirements.txt requirements-llm.txt ./
ARG WITH_LLM=false
RUN pip install -r requirements.txt && if [ "$WITH_LLM" = "true" ]; then pip install -r requirements-llm.txt; fi
COPY app ./app
COPY training ./training
COPY data/kb ./data/kb
COPY data/eval_es_cl.csv ./data/eval_es_cl.csv
# Los modelos entrenados se montan en /app/models (volumen) o se copian desde S3 al iniciar
ENV MODELS_DIR=/app/models KB_PATH=/app/data/kb/telcoaustral_kb.json
RUN useradd -m -u 1000 appuser && mkdir -p /app/models && chown -R appuser /app
USER appuser
EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=3s --start-period=20s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2", "--proxy-headers"]
