FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 \
    CAI_HOME=/app CAI_DATA=/data/app CAI_REPLAY_ONLY=1 PORT=8050 FORWARDED_ALLOW_IPS=127.0.0.1
WORKDIR /app
COPY requirements.lock pyproject.toml README.md LICENSE ./
COPY src ./src
COPY evals ./evals
RUN pip install --no-cache-dir -c requirements.lock . && useradd -u 10001 -m cai && mkdir -p /data/app && chown -R cai /data
USER 10001
EXPOSE 8050
HEALTHCHECK --interval=30s --timeout=3s CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8050/health').status==200 else 1)"
CMD ["sh", "-c", "uvicorn 'cailens.web:create_app' --factory --host 0.0.0.0 --port ${PORT} --proxy-headers --forwarded-allow-ips=${FORWARDED_ALLOW_IPS}"]
