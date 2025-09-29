# app/api.py
import os
from typing import Optional

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Query
from fastapi.responses import JSONResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .logging_setup import setup_logger
log = setup_logger()

from .infer import (
    predict_image, list_classes, get_model,
    validate_and_update_settings, get_settings, ensure_weights,
    get_config_snapshot
)

app = FastAPI(title="Teeth Mask R-CNN Inference API", version="1.3")

# Prepare /files static mount
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RUNTIME_DIR = os.environ.get("RUNTIME_DIR", os.path.join(BASE_DIR, "runtime"))
os.makedirs(RUNTIME_DIR, exist_ok=True)
app.mount("/files", StaticFiles(directory=RUNTIME_DIR), name="files")

# ---------- Schemas ----------
class PredictResponse(BaseModel):
    detections: dict
    images: dict
    meta: dict

class SettingsUpdate(BaseModel):
    weights_path: Optional[str] = Field(default=None, description="Absolute path to weights .h5 inside container")
    return_mask: Optional[bool] = None
    return_masked_image: Optional[bool] = None
    confidence_threshold: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    max_detections: Optional[int] = Field(default=None, ge=0)

# ---------- Lifecycle ----------
@app.on_event("startup")
def _warm():
    get_model()
    log.info("API startup complete")

# ---------- Info ----------
@app.get("/healthz")
def healthz():
    try:
        ensure_weights()
        ready = True
        msg = "ok"
    except Exception as e:
        ready = False
        msg = f"not_ready: {e}"
    return {"status": "ok", "model_ready": ready, "detail": msg}

@app.get("/help")
def help_endpoint():
    return list_classes()

# ---------- Settings ----------
@app.get("/settings")
def read_settings():
    return get_settings()

@app.post("/settings")
def update_settings(payload: SettingsUpdate):
    try:
        merged = validate_and_update_settings(payload.model_dump(exclude_none=True))
        log.info("/settings updated -> %s", merged)
        return {"settings": merged}
    except Exception as e:
        log.exception("Settings update failed")
        raise HTTPException(status_code=400, detail=str(e))

# ---------- Predict ----------
@app.post("/predict", response_model=PredictResponse)
async def predict(
    image: UploadFile = File(...),
    weights_path: Optional[str] = Form(None),
    return_mask: Optional[bool] = Form(False),
    return_masked_image: Optional[bool] = Form(False),
    confidence_threshold: Optional[float] = Form(None),
    max_detections: Optional[int] = Form(None),
):
    if image.content_type is None or not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Please upload an image file.")
    img_bytes = await image.read()
    log.info("/predict file=%s size=%d bytes, opts: weights=%s, mask=%s, masked=%s, thr=%s, max=%s",
             image.filename, len(img_bytes), weights_path, return_mask, return_masked_image,
             confidence_threshold, max_detections)
    try:
        result = predict_image(
            img_bytes,
            weights_path=weights_path,
            return_mask=return_mask,
            return_masked=return_masked_image,
            conf_thresh=confidence_threshold,
            max_det=max_detections,
        )
        return JSONResponse(content=result)
    except FileNotFoundError as fnf:
        log.warning("/predict failed (FileNotFound): %s", fnf)
        raise HTTPException(status_code=400, detail=str(fnf))
    except ValueError as ve:
        log.warning("/predict failed (ValueError): %s", ve)
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        log.exception("/predict failed")
        raise HTTPException(status_code=500, detail=f"Inference error: {e}")

# ---------- Debug endpoints (for troubleshooting only) ----------
@app.get("/debug/config")
def debug_config():
    snap = get_config_snapshot()
    log.info("/debug/config -> %s", snap)
    return snap

@app.get("/debug/verify")
def debug_verify(path: Optional[str] = None):
    """Verify a file path exists (defaults to current weights_path)."""
    p = path or get_settings().get("weights_path", "")
    exists = os.path.isfile(p)
    size = os.path.getsize(p) if exists else None
    log.info("/debug/verify path=%s exists=%s size=%s", p, exists, size)
    return {"path": p, "exists": exists, "size": size}

@app.get("/logs", response_class=PlainTextResponse)
def read_logs(lines: int = Query(200, ge=1, le=5000)):
    """Return the last N lines of the server log."""
    log_path = os.environ.get("LOG_FILE", os.path.join(RUNTIME_DIR, "server.log"))
    if not os.path.exists(log_path):
        return "no logs yet"
    with open(log_path, "rb") as f:
        try:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            # Read up to ~100 bytes per line average
            read_bytes = min(size, lines * 200)
            f.seek(size - read_bytes)
        except Exception:
            f.seek(0)
        tail = f.read().decode(errors="replace")
    return tail