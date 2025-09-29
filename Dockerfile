FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TF_CPP_MIN_LOG_LEVEL=2

# Minimal system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*    


WORKDIR /app

# Install runtime deps
COPY requirements.txt /app/requirements.runtime.txt

# Preinstall numpy so anything legacy that inspects numpy at build-time is safe
RUN pip install --no-cache-dir --upgrade pip setuptools wheel \
    && pip install --no-cache-dir numpy==1.23.5 \
    && pip install --no-cache-dir -r /app/requirements.runtime.txt

# Copy only what’s needed for inference
COPY mrcnn/ /app/mrcnn/
COPY myTools/ /app/myTools/
COPY app/ /app/app/

# Runtime dir for generated files (mask/overlay) and URL prefix
RUN mkdir -p /app/runtime
ENV RUNTIME_DIR=/app/runtime
ENV FILES_URL_PREFIX=/files

EXPOSE 8000
ENV WEIGHTS_PATH=/app/weights/mask_rcnn_teeth.h5
ENV SETTINGS_PATH=/app/app/settings.json
ENV LOGS_DIR=/tmp/logs

# Start uvicorn and Jupyter Lab (notebook dir set to /app/teeth)
# CMD ["bash", "-lc", "uvicorn app.api:app --host 0.0.0.0 --port 8000 & jupyter lab --ip=0.0.0.0 --port=8888 --notebook-dir=/app --no-browser --allow-root --NotebookApp.token=''"]
CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]