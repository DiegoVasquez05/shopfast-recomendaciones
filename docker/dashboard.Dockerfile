FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app
RUN pip install "streamlit>=1.45" "requests>=2.31" "pandas>=2.1"
COPY dashboard/app.py .
RUN useradd -m -u 1000 appuser
USER appuser
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.baseUrlPath=/dashboard", "--browser.gatherUsageStats=false"]
