FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app
COPY requirements.txt requirements-app.lock.txt ./
RUN pip install -r requirements.txt -c requirements-app.lock.txt && useradd --create-home appuser
COPY --chown=appuser:appuser apps ./apps
COPY --chown=appuser:appuser kidsafe ./kidsafe
COPY --chown=appuser:appuser ml ./ml
COPY --chown=appuser:appuser .streamlit ./.streamlit
USER appuser
CMD ["python", "-m", "uvicorn", "apps.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--limit-concurrency", "16"]
