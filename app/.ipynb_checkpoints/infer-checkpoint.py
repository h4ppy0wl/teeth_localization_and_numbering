# app/infer.py
import os
import io
import json
import time
import threading
from typing import Dict, Any, List, Optional, Tuple
import uuid

import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Make local packages importable
APP_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
import sys
sys.path.append(APP_DIR)

# Logging
from .logging_setup import setup_logger
log = setup_logger()

# Import your TF2 Matterport stack
from mrcnn import model as modellib
from mrcnn.config import Config

# Your preprocessing
from myTools.digileap_preprocessing import dental_gray_world_white_balance

# ---------- Class map (filtered) ----------
FILTERED_CLASS_DICTIONARY = {
    'T11': 1, 'T12': 2, 'T13': 3, 'T14': 4, 'T15': 5, 'T16': 6, 'T17': 7,
    'T21': 8, 'T22': 9, 'T23': 10, 'T24': 11, 'T25': 12, 'T26': 13, 'T27': 14,
    'T31': 15, 'T32': 16, 'T33': 17, 'T34': 18, 'T35': 19, 'T36': 20, 'T37': 21,
    'T41': 22, 'T42': 23, 'T43': 24, 'T44': 25, 'T45': 26, 'T46': 27, 'T47': 28
}
ID_TO_NAME = {v: k for k, v in FILTERED_CLASS_DICTIONARY.items()}

# ---------- Config that mirrors your InferenceConfig ----------
class BaseTeethConfig(Config):
    NAME = "teeth"
    IMAGES_PER_GPU = 1
    BACKBONE = "resnet50"
    NUM_CLASSES = 1 + len(FILTERED_CLASS_DICTIONARY)
    USE_MINI_MASK = False
    IMAGE_RESIZE_MODE = "square"
    IMAGE_MIN_DIM = 1024
    IMAGE_MAX_DIM = 1024
    DETECTION_MIN_CONFIDENCE = 0.5
    DETECTION_MAX_INSTANCES = 1
    DETECTION_MAX_INSTANCES_ALL_CLASSES = 30
    DETECTION_NMS_THRESHOLD = 0.1
    DETECTION_NMS_THRESHOLD_ALL_CLASSES = 0.7

class InferenceConfig(BaseTeethConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

# ---------- Settings ----------
DEFAULT_SETTINGS = {
    "weights_path": "",               # absolute path mounted inside container
    "return_mask": False,
    "return_masked_image": False,
    "confidence_threshold": 0.5,
    "max_detections": 30
}

SETTINGS_PATH = os.environ.get("SETTINGS_PATH", os.path.join(APP_DIR, "app", "settings.json"))

def _load_json(path: str) -> Dict[str, Any]:
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        return {}

def _dump_json(path: str, data: Dict[str, Any]) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2, sort_keys=True)
    os.replace(tmp, path)

def load_settings() -> Dict[str, Any]:
    on_disk = _load_json(SETTINGS_PATH)
    out = {**DEFAULT_SETTINGS, **on_disk}
    log.debug("load_settings -> %s", out)
    return out

def save_settings(new_settings: Dict[str, Any]) -> Dict[str, Any]:
    merged = {**load_settings(), **new_settings}
    _dump_json(SETTINGS_PATH, merged)
    log.info("save_settings: updated settings.json with %s", new_settings)
    return merged

# Where we save files and how they’re served
RUNTIME_DIR = os.environ.get("RUNTIME_DIR", os.path.join(APP_DIR, "runtime"))
FILES_URL_PREFIX = os.environ.get("FILES_URL_PREFIX", "/files")
os.makedirs(RUNTIME_DIR, exist_ok=True)

# ---------- Singleton model loader + weight hot-swap ----------
_MODEL = None
_CONFIG = None
_CURRENT_WEIGHTS_PATH = None
_LOAD_LOCK = threading.Lock()

def get_model():
    """Create the Mask R-CNN model once; weights are hot-swapped via ensure_weights()."""
    global _MODEL, _CONFIG
    if _MODEL is not None:
        return _MODEL, _CONFIG
    with _LOAD_LOCK:
        if _MODEL is not None:
            return _MODEL, _CONFIG
        _CONFIG = InferenceConfig()
        log.info("Creating inference model with config: "
                 "MIN_CONF=%.3f, MAX_INST=%d, MAX_ALL=%d, IMG_[%d..%d], NMS=%.2f/%.2f",
                 _CONFIG.DETECTION_MIN_CONFIDENCE,
                 _CONFIG.DETECTION_MAX_INSTANCES,
                 getattr(_CONFIG, "DETECTION_MAX_INSTANCES_ALL_CLASSES", -1),
                 _CONFIG.IMAGE_MIN_DIM, _CONFIG.IMAGE_MAX_DIM,
                 _CONFIG.DETECTION_NMS_THRESHOLD,
                 getattr(_CONFIG, "DETECTION_NMS_THRESHOLD_ALL_CLASSES", -1))
        logs_dir = os.environ.get("LOGS_DIR", "/tmp/logs")
        os.makedirs(logs_dir, exist_ok=True)
        model = modellib.MaskRCNN(mode="inference", config=_CONFIG, model_dir=logs_dir)
        _MODEL = model
    return _MODEL, _CONFIG

def ensure_weights():
    """
    Ensure the model has the weights pointed to by settings['weights_path'].
    If changed since last load, (re)load them.
    """
    global _CURRENT_WEIGHTS_PATH
    model, _ = get_model()
    settings = load_settings()
    desired = settings.get("weights_path", "").strip()

    if not desired:
        log.error("ensure_weights: weights_path not set in settings.json")
        raise FileNotFoundError("No weights_path configured. Set it via POST /settings or send with /predict.")
    if not os.path.isfile(desired):
        log.error("ensure_weights: weights not found at %s", desired)
        raise FileNotFoundError(f"weights_path does not exist: {desired}")

    # Only reload if path changed
    if _CURRENT_WEIGHTS_PATH != desired:
        with _LOAD_LOCK:
            if _CURRENT_WEIGHTS_PATH != desired:
                t0 = time.perf_counter()
                log.info("Loading weights from %s ...", desired)
                model.load_weights(desired, by_name=True)
                dt = (time.perf_counter() - t0) * 1000
                _CURRENT_WEIGHTS_PATH = desired
                log.info("Weights loaded in %.1f ms", dt)
    else:
        log.debug("ensure_weights: already using %s", desired)

# ---------- Utilities for masks/labels ----------
def _img_to_array(file_bytes: bytes) -> np.ndarray:
    with Image.open(io.BytesIO(file_bytes)) as im:
        im = im.convert("RGB")
        arr = np.array(im)
    log.debug("Image loaded: shape=%s dtype=%s min=%s max=%s",
              arr.shape, arr.dtype, arr.min(), arr.max())
    # apply same preprocessing as training
    arr = dental_gray_world_white_balance(arr)
    log.debug("After preprocessing: shape=%s dtype=%s min=%s max=%s",
              arr.shape, arr.dtype, arr.min(), arr.max())
    return arr

def _array_to_pil(arr: np.ndarray) -> Image.Image:
    return Image.fromarray(arr)

def _assign_neighbor_distinct_colors(masks: np.ndarray) -> List[Tuple[int, int, int]]:
    if masks is None or masks.size == 0:
        return []
    from skimage.morphology import binary_dilation
    H, W, N = masks.shape
    adj = [set() for _ in range(N)]
    se = np.array([[0,1,0],[1,1,1],[0,1,0]], dtype=bool)
    for i in range(N):
        mi = masks[:,:,i].astype(bool)
        di = binary_dilation(mi, se)
        for j in range(i+1, N):
            mj = masks[:,:,j].astype(bool)
            if (mi & mj).any() or (di & mj).any():
                adj[i].add(j); adj[j].add(i)
    palette = [
        (230, 25, 75),(60, 180, 75),(255, 225, 25),(0, 130, 200),
        (245, 130, 48),(145, 30, 180),(70, 240, 240),(240, 50, 230),
        (210, 245, 60),(250, 190, 190),(0, 128, 128),(230, 190, 255),
        (170, 110, 40),(255, 250, 200),(128, 0, 0),(170, 255, 195),
        (128, 128, 0),(255, 215, 180),(0, 0, 128),(128, 128, 128)
    ]
    colors_idx = [-1]*N
    for i in range(N):
        forbidden = {colors_idx[n] for n in adj[i] if colors_idx[n] != -1}
        for c in range(len(palette)):
            if c not in forbidden:
                colors_idx[i] = c
                break
        if colors_idx[i] == -1:
            colors_idx[i] = 0
    return [palette[c % len(palette)] for c in colors_idx]

def _label_positions_top_left(masks: np.ndarray) -> List[Tuple[int, int]]:
    if masks is None or masks.size == 0:
        return []
    positions: List[Tuple[int, int]] = []
    for i in range(masks.shape[-1]):
        m = masks[:, :, i]
        ys, xs = np.nonzero(m)
        if ys.size == 0:
            positions.append((0, 0))
        else:
            y_min, x_min = int(ys.min()), int(xs.min())
            positions.append((x_min, y_min))
    return positions

def _make_mask_canvas_rgba(masks: np.ndarray, class_names: List[str]) -> np.ndarray:
    if masks is None or masks.size == 0:
        return np.zeros((32, 32, 4), dtype=np.uint8)
    H, W, N = masks.shape
    canvas = np.zeros((H, W, 4), dtype=np.uint8)
    colors = _assign_neighbor_distinct_colors(masks)

    for i in range(N):
        m = masks[:, :, i].astype(bool)
        if not m.any():
            continue
        r, g, b = colors[i]
        canvas[m, 0] = r
        canvas[m, 1] = g
        canvas[m, 2] = b
        canvas[m, 3] = 255

    img = Image.fromarray(canvas, mode="RGBA")
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()
    positions = _label_positions_top_left(masks)
    for i, (x, y) in enumerate(positions):
        label = class_names[i] if i < len(class_names) else "obj"
        draw.text((x + 2, y + 2), label, fill=(255, 255, 255, 255),
                  font=font, anchor="la", stroke_width=2, stroke_fill=(0, 0, 0, 255))
    return np.array(img)

def _make_mask_overlay_with_labels(image: np.ndarray, masks: np.ndarray,
                                   class_names: List[str], alpha: float = 0.5) -> np.ndarray:
    out = image.copy().astype(np.float32)
    if masks is None or masks.size == 0:
        return image
    N = masks.shape[-1]
    colors = _assign_neighbor_distinct_colors(masks)
    for i in range(N):
        m = masks[:, :, i].astype(bool)
        if not m.any():
            continue
        color = np.array(colors[i], dtype=np.float32)
        out[m] = (1 - alpha) * out[m] + alpha * color
    out = np.clip(out, 0, 255).astype(np.uint8)

    img = Image.fromarray(out)
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()
    positions = _label_positions_top_left(masks)
    for i, (x, y) in enumerate(positions):
        label = class_names[i] if i < len(class_names) else "obj"
        draw.text((x + 2, y + 2), label, fill=(255, 255, 255),
                  font=font, anchor="la", stroke_width=2, stroke_fill=(0, 0, 0))
    return np.array(img)

def _save_image(arr: np.ndarray, filename: str) -> str:
    path = os.path.join(RUNTIME_DIR, filename)
    ext = os.path.splitext(filename)[1].lower()
    img = Image.fromarray(arr)
    if ext in (".jpg", ".jpeg"):
        img = img.convert("RGB")
        img.save(path, format="JPEG", quality=90)
    else:
        img.save(path, format="PNG")
    url = f"{FILES_URL_PREFIX}/{filename}"
    log.info("Saved image: %s -> %s", path, url)
    return url

# ---------- Public API ----------
def validate_and_update_settings(partial: Dict[str, Any]) -> Dict[str, Any]:
    cleaned: Dict[str, Any] = {}
    if "weights_path" in partial:
        wp = str(partial["weights_path"]).strip()
        if not wp:
            raise ValueError("weights_path cannot be empty.")
        if not os.path.isabs(wp):
            raise ValueError("weights_path must be an absolute path inside the container.")
        cleaned["weights_path"] = wp

    if "return_mask" in partial:
        cleaned["return_mask"] = bool(partial["return_mask"])
    if "return_masked_image" in partial:
        cleaned["return_masked_image"] = bool(partial["return_masked_image"])

    if "confidence_threshold" in partial and partial["confidence_threshold"] is not None:
        ct = float(partial["confidence_threshold"])
        cleaned["confidence_threshold"] = max(0.0, min(1.0, ct))
    if "max_detections" in partial and partial["max_detections"] is not None:
        md = int(partial["max_detections"])
        cleaned["max_detections"] = max(0, md)

    merged = save_settings(cleaned)
    return merged

def predict_image(file_bytes: bytes,
                  weights_path: Optional[str] = None,
                  return_mask: Optional[bool] = None,
                  return_masked: Optional[bool] = None,
                  conf_thresh: Optional[float] = None,
                  max_det: Optional[int] = None) -> Dict[str, Any]:
    # Persist overrides (if provided)
    overrides: Dict[str, Any] = {}
    if weights_path is not None: overrides["weights_path"] = weights_path
    if return_mask is not None: overrides["return_mask"] = return_mask
    if return_masked is not None: overrides["return_masked_image"] = return_masked
    if conf_thresh is not None: overrides["confidence_threshold"] = conf_thresh
    if max_det is not None: overrides["max_detections"] = max_det
    if overrides:
        log.info("predict_image overrides -> %s", overrides)
        validate_and_update_settings(overrides)

    # Ensure weights loaded
    ensure_weights()
    settings = load_settings()

    # Image -> array (+ preprocess)
    image = _img_to_array(file_bytes)

    # Detect
    model, cfg = get_model()
    t0 = time.perf_counter()
    result = model.detect([image], verbose=0)[0]
    dt_detect = (time.perf_counter() - t0) * 1000
    log.info("detect() done in %.1f ms. Raw: rois=%s, masks=%s, scores=%s",
             dt_detect,
             None if result.get("rois") is None else result["rois"].shape,
             None if result.get("masks") is None else result["masks"].shape,
             None if result.get("scores") is None else len(result["scores"]))

    # Filter by confidence & truncate
    keep = [i for i, s in enumerate(result["scores"]) if s >= settings["confidence_threshold"]] if result.get("scores") is not None else []
    if len(keep) > settings["max_detections"]:
        keep = keep[:settings["max_detections"]]
    log.info("Filtering: threshold=%.3f, max_det=%d -> keep=%d",
             settings["confidence_threshold"], settings["max_detections"], len(keep))

    rois   = result["rois"][keep] if len(keep) else np.zeros((0,4), dtype=np.int32)
    scores = result["scores"][keep] if len(keep) else np.zeros((0,), dtype=np.float32)
    cids   = result["class_ids"][keep] if len(keep) else np.zeros((0,), dtype=np.int32)
    masks  = result["masks"][:, :, keep] if (len(keep) and result.get("masks") is not None) else np.zeros((image.shape[0], image.shape[1], 0), dtype=bool)

    class_ids = cids.tolist()
    class_names = [ID_TO_NAME.get(int(i), f"ID_{int(i)}") for i in class_ids]
    scores_list = [float(s) for s in scores]
    boxes = rois.tolist()
    log.debug("Post-filter: boxes=%d, masks=%s, classes=%s", len(boxes), masks.shape, class_names)

    # Optional files
    mask_url = None
    masked_url = None
    if settings["return_mask"]:
        mask_rgba = _make_mask_canvas_rgba(masks, class_names) if masks.shape[-1] > 0 else np.zeros((image.shape[0], image.shape[1], 4), dtype=np.uint8)
        mask_url = _save_image(mask_rgba, f"{uuid.uuid4().hex}_mask.png")
    if settings["return_masked_image"]:
        overlay = _make_mask_overlay_with_labels(image, masks, class_names, alpha=0.5)
        masked_url = _save_image(overlay, f"{uuid.uuid4().hex}_overlay.jpg")

    out = {
        "detections": {
            "boxes_yxyx": boxes,
            "scores": scores_list,
            "class_ids": class_ids,
            "class_names": class_names
        },
        "images": {
            "mask_url": mask_url,
            "masked_url": masked_url
        },
        "meta": {
            "num_detections": len(class_ids),
            "confidence_threshold": settings["confidence_threshold"],
            "max_detections": settings["max_detections"],
            "weights_path": settings["weights_path"]
        }
    }
    log.info("predict_image -> num_detections=%d", out["meta"]["num_detections"])
    return out

def list_classes() -> Dict[str, Any]:
    return {
        "num_classes": len(FILTERED_CLASS_DICTIONARY),
        "classes": [{"id": i, "name": n} for n, i in FILTERED_CLASS_DICTIONARY.items()]
    }

def get_settings() -> Dict[str, Any]:
    return load_settings()

def get_config_snapshot() -> Dict[str, Any]:
    _, cfg = get_model()
    return {
        "IMAGE_MIN_DIM": cfg.IMAGE_MIN_DIM,
        "IMAGE_MAX_DIM": cfg.IMAGE_MAX_DIM,
        "IMAGE_RESIZE_MODE": cfg.IMAGE_RESIZE_MODE,
        "DETECTION_MIN_CONFIDENCE": cfg.DETECTION_MIN_CONFIDENCE,
        "DETECTION_MAX_INSTANCES": cfg.DETECTION_MAX_INSTANCES,
        "DETECTION_MAX_INSTANCES_ALL_CLASSES": getattr(cfg, "DETECTION_MAX_INSTANCES_ALL_CLASSES", None),
        "DETECTION_NMS_THRESHOLD": cfg.DETECTION_NMS_THRESHOLD,
        "DETECTION_NMS_THRESHOLD_ALL_CLASSES": getattr(cfg, "DETECTION_NMS_THRESHOLD_ALL_CLASSES", None),
        "NUM_CLASSES": cfg.NUM_CLASSES,
        "BACKBONE": cfg.BACKBONE
    }