# "Will You Get In?" prediction service.
# The model is NOT baked into the image: mount the directory made by
# scripts/package_model.sh at /app/service/model_files (see compose.yaml).
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app
COPY service/requirements.txt service/requirements.txt
RUN pip install --no-cache-dir -r service/requirements.txt

COPY service/ service/

RUN useradd --system --uid 10001 app
USER app

EXPOSE 8000
# One worker keeps a single copy of the model in memory; two threads let a
# health check or second visitor through while a prediction runs.
CMD ["gunicorn", "--workers", "1", "--threads", "2", "--bind", "0.0.0.0:8000", \
     "--access-logfile", "-", "--timeout", "30", "service.wsgi:app"]

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4)"
