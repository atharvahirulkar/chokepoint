# Combined Docker image for Hugging Face Spaces.
#   - FastAPI backend on :8000 (internal)
#   - Streamlit dashboard on :8501 (exposed publicly via HF Spaces)
# Both processes start from /app/start.sh; uvicorn runs in the background
# and Streamlit runs in the foreground so the container stays alive.

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    CHOKEPOINT_API=http://localhost:8000 \
    # HF Spaces sometimes runs as a non-root user; ensure home is writable
    HOME=/tmp \
    XDG_CACHE_HOME=/tmp/.cache \
    STREAMLIT_CONFIG_DIR=/tmp/.streamlit \
    MPLCONFIGDIR=/tmp/.mpl

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy code + processed artifacts. HF Space repo will have these committed.
COPY pipeline/ pipeline/
COPY models/ models/
COPY api/ api/
COPY dashboard/ dashboard/
COPY data/processed/ data/processed/
COPY data/synthetic/ data/synthetic/
COPY eval_report.json ./eval_report.json
COPY deploy/hf_space/start.sh /start.sh

RUN chmod +x /start.sh && \
    mkdir -p /tmp/.cache /tmp/.streamlit /tmp/.mpl && \
    chmod -R 777 /tmp

EXPOSE 8501

CMD ["/start.sh"]
