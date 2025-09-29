# app/infer.py
import os
import json
import time
import threading
import uuid
import tempfile
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
from PIL import Image

# Make local packages importable
APP_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
import sys
if APP_DIR not in sys.path:
    sys.path.append(APP_DIR)

# --- logging ---------------------------------------------------------------
import logging
log = logging.getLogger("infer")
if not log.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s %(name)s: %(message)s"))
    log.addHandler(_h)
log.setLevel(os.environ.get("LOGLEVEL", "INFO").upper())

# --- external libs ---------------------------------------------------------
from mrcnn import model as modellib
from mrcnn.config import Config
from mrcnn import visualize

# Your base utilities (dataset, config, preprocessing)
import myTools.teeth_detection_base as tdb

# --- defaults & settings ---------------------------------------------------
DEFAULT_SETTINGS = {
    "weights_path": "",               # absolute path inside container
    "return_mask": False,
    "return_masked_image": False,
    "confidence_threshold": 0.5,      # informational only
    "max_detections": 30              # informational only
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
    os.makedirs(os.path.dirname(SETTINGS_PATH), exist_ok=True)
    _dump_json(SETTINGS_PATH, merged)
    return merged

# Exposed for API: GET /settings
def get_settings() -> Dict[str, Any]:
    return load_settings()

# --- config & model --------------------------------------------------------
class TeethInferenceConfig(tdb.TeethConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

_CONFIG: Optional[Config] = None
_MODEL: Optional[modellib.MaskRCNN] = None
_CURRENT_WEIGHTS_PATH: Optional[str] = None
_LOAD_LOCK = threading.Lock()

def get_model() -> Tuple[modellib.MaskRCNN, Config]:
    """Create a singleton inference model and config."""
    global _MODEL, _CONFIG
    if _CONFIG is None:
        _CONFIG = TeethInferenceConfig()
        _CONFIG.display() if hasattr(_CONFIG, "display") else None
        log.info("Config prepared. NUM_CLASSES=%s, BACKBONE=%s", _CONFIG.NUM_CLASSES, getattr(_CONFIG, "BACKBONE", "?"))
    if _MODEL is None:
        logs_dir = os.path.join(APP_DIR, "logs")
        os.makedirs(logs_dir, exist_ok=True)
        _MODEL = modellib.MaskRCNN(mode="inference", config=_CONFIG, model_dir=logs_dir)
        log.info("MaskRCNN inference model instantiated.")
    return _MODEL, _CONFIG

def ensure_weights():
    """(Re)load weights if settings changed (expects weights mounted inside container)."""
    global _CURRENT_WEIGHTS_PATH
    model, _ = get_model()
    settings = load_settings()
    desired = settings.get("weights_path", "").strip()
    if not desired:
        log.error("ensure_weights: weights_path not set")
        raise FileNotFoundError("No weights_path configured.")
    if not os.path.isfile(desired):
        log.error("ensure_weights: weights not found at %s", desired)
        raise FileNotFoundError(f"Weights not found: {desired}")
    if _CURRENT_WEIGHTS_PATH != desired:
        with _LOAD_LOCK:
            if _CURRENT_WEIGHTS_PATH != desired:
                t0 = time.perf_counter()
                log.info("Loading weights from %s ...", desired)
                model.load_weights(desired, by_name=True)
                _CURRENT_WEIGHTS_PATH = desired
                log.info("Weights loaded in %.1f ms", (time.perf_counter() - t0) * 1000)
    else:
        log.debug("ensure_weights: already using %s", desired)

# --- files / artifacts -----------------------------------------------------
# Make artifacts land where api.py serves them from: /files -> RUNTIME_DIR
RUNTIME_DIR = os.environ.get("RUNTIME_DIR", os.path.join(APP_DIR, "runtime"))
os.makedirs(RUNTIME_DIR, exist_ok=True)

def _save_image(arr: np.ndarray, filename: str) -> str:
    """Save numpy image array to RUNTIME_DIR and return a /files/... URL."""
    path = os.path.join(RUNTIME_DIR, filename)
    Image.fromarray(arr).save(path)
    rel = f"/files/{filename}"
    log.debug("Saved image %s -> %s", filename, rel)
    return rel

def _write_temp_image(file_bytes: bytes, suffix: str = ".jpg") -> str:
    """Persist uploaded bytes to a temp file so we can use tdb.SingleImageDataset."""
    fd, tmp_path = tempfile.mkstemp(suffix=suffix, prefix="upload_")
    with os.fdopen(fd, "wb") as f:
        f.write(file_bytes)
    log.info("Wrote temp image: %s (%d bytes)", tmp_path, len(file_bytes))
    return tmp_path

# --- visualize helpers (mirror notebook look) -----------------------------
def _render_overlay_with_display_instances(image_rgb: np.ndarray,
                                           rois: np.ndarray,
                                           masks: np.ndarray,
                                           class_ids: np.ndarray,
                                           class_names: List[str],
                                           scores: Optional[np.ndarray]) -> np.ndarray:
    """
    Render the classic Matterport overlay (bboxes + labels + colored masks)
    to an off-screen buffer with *no borders* (tight to the image).
    Works with visualize.py versions that do or don’t accept 'ax'.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    H, W = image_rgb.shape[:2]
    inch = 100.0  # 100 dpi → inches = pixels/100
    try:
        # Preferred: versions that accept 'ax'
        fig = plt.figure(figsize=(W / inch, H / inch), dpi=100)
        ax = plt.axes([0, 0, 1, 1])   # fill figure
        ax.axis("off")
        visualize.display_instances(
            image_rgb, rois, masks, class_ids, class_names,
            scores=scores, ax=ax, show_mask=True, show_bbox=True
            # fontsize=14,          # make text bigger
            # fontcolor="k"         # 'k' = black
        )
        # hard-trim any layout padding
        fig.subplots_adjust(0, 0, 1, 1)
        fig.patch.set_alpha(0.0)
        ax.set_xlim(0, W)
        ax.set_ylim(H, 0)
    except TypeError:
        # Fallback: no 'ax' arg available — draw with the function's own figure
        plt.close("all")
        visualize.display_instances(
            image_rgb, rois, masks, class_ids, class_names,
            scores=scores, show_mask=True, show_bbox=True,
            figsize=(W / inch, H / inch)  # many versions accept figsize
            # fontsize=14,          # make text bigger
            # fontcolor="k"         # 'k' = black
        )
        fig = plt.gcf()
        ax = plt.gca()
        # force the canvas to match the image exactly and remove padding
        fig.set_size_inches(W / inch, H / inch)
        fig.set_dpi(100)
        fig.subplots_adjust(0, 0, 1, 1)
        try:
            ax.set_position([0, 0, 1, 1])
        except Exception:
            pass
        for spine in getattr(ax, "spines", {}).values():
            spine.set_visible(False)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.axis("off")
        fig.patch.set_alpha(0.0)
        ax.set_xlim(0, W)
        ax.set_ylim(H, 0)

    # Read pixels back
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8).reshape((h, w, 3))
    plt.close(fig)
    return buf

def _build_masked_image_with_apply_mask(image_rgb: np.ndarray,
                                        masks: np.ndarray) -> np.ndarray:
    """Blend masks using visualize.apply_mask repeatedly (notebook style)."""
    out = image_rgb.copy()
    rng = np.random.RandomState(1234)
    for i in range(masks.shape[-1]):
        color = rng.rand(3)  # [0,1]
        out = visualize.apply_mask(out, masks[:, :, i], color, alpha=0.5)
    return out

def _build_union_mask_rgba(masks: np.ndarray, H: int, W: int) -> np.ndarray:
    """
    Build a transparent PNG mask (RGBA) where alpha=255 on any instance mask,
    alpha=0 elsewhere. RGB is white on masked pixels.
    """
    out = np.zeros((H, W, 4), dtype=np.uint8)  # default fully transparent
    if masks.size == 0:
        return out
    union = masks.any(axis=-1)  # HxW boolean
    out[union, :3] = 255  # white RGB on masked area (can be any color; alpha does the job)
    out[union, 3] = 255   # opaque where masked
    return out


# --- polygons from masks ----------------------------------------------------
def _mask_to_polygons(mask_2d: np.ndarray,
                      approx_tolerance: float = 2.0,
                      min_vertices: int = 3) -> List[List[List[int]]]:
    """
    Convert a single boolean mask (H, W) to one or more polygons.
    Returns: [ [ [x,y], [x,y], ... ],  ... ]  # one polygon per ring
    Coordinates are integer pixel coords in image space.
    """
    from skimage.measure import find_contours, approximate_polygon

    H, W = mask_2d.shape[:2]
    polys: List[List[List[int]]] = []
    # find_contours expects 1 for mask; 0.5 traces boundary between 0 and 1
    contours = find_contours(mask_2d.astype(np.uint8), level=0.5)

    for contour in contours:
        # contour is array of (row, col) floats
        # approximate to reduce points and then map to int pixel coords
        approx = approximate_polygon(contour, tolerance=approx_tolerance)
        if approx.shape[0] < min_vertices:
            continue
        # convert to [x, y] int list and clip to image bounds
        ring = []
        for (y, x) in approx:
            xi = int(round(x))
            yi = int(round(y))
            if xi < 0: xi = 0
            if yi < 0: yi = 0
            if xi >= W: xi = W - 1
            if yi >= H: yi = H - 1
            ring.append([xi, yi])
        # ensure ring is closed (first == last)
        if ring and ring[0] != ring[-1]:
            ring.append(ring[0])
        if len(ring) >= min_vertices + 1:
            polys.append(ring)
    return polys


def _masks_to_polygons_xy(masks: np.ndarray,
                          approx_tolerance: float = 2.0) -> List[Dict[str, List[int]]]:
    """
    Convert masks (H, W, N) into polygons with xs/ys arrays.
    Returns: list of dicts, one per detection: {"xs": [...], "ys": [...]}
    Assumes each mask has a single external ring.
    """
    from skimage.measure import find_contours, approximate_polygon

    if masks.size == 0:
        return []

    H, W, N = masks.shape
    out: List[Dict[str, List[int]]] = []
    for i in range(N):
        mask = masks[:, :, i].astype(np.uint8)
        contours = find_contours(mask, level=0.5)
        if not contours:
            out.append({"xs": [], "ys": []})
            continue
        # take the longest contour
        contour = max(contours, key=len)
        approx = approximate_polygon(contour, tolerance=approx_tolerance)
        xs, ys = [], []
        for (y, x) in approx:
            xi = int(round(x))
            yi = int(round(y))
            if 0 <= xi < W and 0 <= yi < H:
                xs.append(xi)
                ys.append(yi)
        # close the polygon (repeat first point)
        if xs and ys and (xs[0] != xs[-1] or ys[0] != ys[-1]):
            xs.append(xs[0])
            ys.append(ys[0])
        out.append({"xs": xs, "ys": ys})
    return out


# --- public api ------------------------------------------------------------
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
        cleaned["confidence_threshold"] = float(partial["confidence_threshold"])
    if "max_detections" in partial and partial["max_detections"] is not None:
        cleaned["max_detections"] = int(partial["max_detections"])

    merged = save_settings(cleaned)
    return merged

# Exposed for API: /help
def list_classes() -> Dict[str, Any]:
    """Return class info consistent with current config/dataset utilities."""
    _, cfg = get_model()
    # Build an ordered list of class names like dataset.class_names would: BG + keys()
    class_names = ["BG"] + list(getattr(cfg, "teeth_classes", {}).keys())
    id_to_name = {i: name for i, name in enumerate(class_names)}
    return {
        "num_classes": len(class_names),
        "class_names": class_names,
        "id_to_name": id_to_name,
    }

def predict_image(file_bytes: bytes,
                  weights_path: Optional[str] = None,
                  return_mask: Optional[bool] = None,
                  return_masked: Optional[bool] = None,
                  conf_thresh: Optional[float] = None,
                  max_det: Optional[int] = None) -> Dict[str, Any]:
    """
    Notebook-like prediction path (no extra post-filtering):
      1) ensure weights loaded
      2) write uploaded bytes to a temp file
      3) build SingleImageDataset (adds classes from config.teeth_classes)
      4) ds.load_image(0) -> preprocessing occurs
      5) model.detect([img])
      6) return raw outputs; overlays/masks via mrcnn.visualize
    """
    # merge runtime overrides with persisted settings (kept for compatibility)
    settings = load_settings()
    if weights_path:
        settings["weights_path"] = weights_path
    if return_mask is not None:
        settings["return_mask"] = bool(return_mask)
    if return_masked is not None:
        settings["return_masked_image"] = bool(return_masked)
    if conf_thresh is not None:
        settings["confidence_threshold"] = float(conf_thresh)
    if max_det is not None:
        settings["max_detections"] = int(max_det)
    # persist only weights_path (so ensure_weights sees it)
    save_settings({"weights_path": settings["weights_path"]})
    ensure_weights()
    model, cfg = get_model()

    # Build dataset as in the notebook
    tmp_img_path = _write_temp_image(file_bytes, suffix=".jpg")
    try:
        ds = tdb.SingleImageDataset()
        class_names_cfg = list(getattr(cfg, "teeth_classes", {}).keys())
        ds.add_classes(class_names=class_names_cfg, source="teeth")
        ds.add_image_from_path(image_path=tmp_img_path, image_id=0, source="teeth")
        ds.prepare()
        img = ds.load_image(0)  # includes preprocessing in base file
        log.info("Dataset prepared. class_names=%s", ds.class_names)
        log.info("Image loaded via dataset. shape=%s dtype=%s", getattr(img, "shape", None), getattr(img, "dtype", None))

        t0 = time.perf_counter()
        result = model.detect([img], verbose=0)[0]
        dt_detect = (time.perf_counter() - t0) * 1000
        log.info("detect() done in %.1f ms. rois=%s masks=%s scores=%s",
                 dt_detect,
                 None if result.get("rois") is None else result["rois"].shape,
                 None if result.get("masks") is None else result["masks"].shape,
                 None if result.get("scores") is None else len(result["scores"]))
    finally:
        try:
            os.remove(tmp_img_path)
            log.debug("Temp image removed: %s", tmp_img_path)
        except Exception as e:
            log.warning("Failed to remove temp image %s: %s", tmp_img_path, e)

    # Direct outputs (no extra filtering)
    rois   = result.get("rois",   np.zeros((0, 4), dtype=np.int32))
    masks  = result.get("masks",  np.zeros((img.shape[0], img.shape[1], 0), dtype=bool))
    scores = result.get("scores", np.zeros((0,), dtype=np.float32))
    cids   = result.get("class_ids", np.zeros((0,), dtype=np.int32))
    # Polygons per instance (aligned with masks order)
    polygons_xy = _masks_to_polygons_xy(masks, approx_tolerance=2.0)

    # Map ids -> names like the notebook: use dataset.class_names (BG + your keys)
    boxes_yxyx = rois.tolist()
    class_ids = cids.tolist()
    scores_list = scores.tolist()
    class_names_per_det = [ds.class_names[int(i)] if 0 <= int(i) < len(ds.class_names) else f"ID_{int(i)}"
                           for i in class_ids]

    # Optional renders via Mask R-CNN visualize helpers
   # Optional renders via Mask R-CNN visualize helpers
    mask_url = None
    masked_url = None

    # Dimensions for renders
    H, W = img.shape[:2]

    uuid_temp = uuid.uuid4().hex
    # 1) Transparent PNG of the union mask -> mask_url
    if settings["return_mask"] and masks.shape[-1] > 0:
        union_rgba = _build_union_mask_rgba(masks, H, W)
        mask_url = _save_image(union_rgba, f"{uuid_temp}_union_mask.png")

    # 2) Notebook-style overlay with boxes + labels + colored masks -> masked_url
    if settings["return_masked_image"] and (masks.shape[-1] > 0 or rois.shape[0] > 0):
        overlay = _render_overlay_with_display_instances(
            image_rgb=img,
            rois=rois,
            masks=masks,
            class_ids=cids,
            class_names=ds.class_names,
            scores=scores
        )
        masked_url = _save_image(overlay, f"{uuid_temp}_overlay.jpg")


    # Also save the JSON result and provide a URL for it
    json_path = os.path.join(RUNTIME_DIR, f"{uuid_temp}_result.json")
    json_url = f"/files/{uuid_temp}_result.json"
    
    # Construct the final result dictionary
    result_data = {
        "detections": {
            "boxes_yxyx": boxes_yxyx,
            "scores": scores_list,
            "class_ids": class_ids,
            "class_names": class_names_per_det,
            "polygons": polygons_xy   # NEW format: {"xs": [...], "ys": [...]}
        },
        "images": {
            "mask_url": mask_url,
            "masked_url": masked_url,
            "json_url": json_url
        },
        "meta": {
            "num_detections": len(boxes_yxyx),
            "confidence_threshold": settings["confidence_threshold"],
            "max_detections": settings["max_detections"],
            "weights_path": _CURRENT_WEIGHTS_PATH
        }
    }



    log.info("Saving prediction result to %s (URL: %s)", json_path, json_url)
    _dump_json(json_path, result_data)

    return result_data

def get_config_snapshot() -> Dict[str, Any]:
    _, cfg = get_model()
    return {
        "IMAGE_MIN_DIM": getattr(cfg, "IMAGE_MIN_DIM", None),
        "IMAGE_MAX_DIM": getattr(cfg, "IMAGE_MAX_DIM", None),
        "IMAGE_RESIZE_MODE": getattr(cfg, "IMAGE_RESIZE_MODE", None),
        "DETECTION_MIN_CONFIDENCE": getattr(cfg, "DETECTION_MIN_CONFIDENCE", None),
        "DETECTION_MAX_INSTANCES": getattr(cfg, "DETECTION_MAX_INSTANCES", None),
        "DETECTION_NMS_THRESHOLD": getattr(cfg, "DETECTION_NMS_THRESHOLD", None),
        "NUM_CLASSES": getattr(cfg, "NUM_CLASSES", None),
        "BACKBONE": getattr(cfg, "BACKBONE", None)
    }
