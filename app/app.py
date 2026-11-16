import os
import random
import shutil
import tempfile
import time
import traceback

import numpy as np
import rasterio
import rasterio.errors
import streamlit as st
from rasterio.enums import Resampling

from core.components import render_hero
from core.config import PENGUIN_FACTS
from core.styles import apply_custom_styles
from models.model_registry import list_models, load, load_config
from pipeline.counter import compute_metrics
from pipeline.gsd import gsd_from_altitude, gsd_from_metadata
from pipeline.inference import run as run_inference
from pipeline.nms import apply_nms
from utils.exporter import to_label_studio_json, to_roi_zip
from utils.visualizer import draw_boxes, to_bytes


@st.cache_resource(show_spinner=False)
def get_model(name: str):
    return load(name)


@st.cache_data(show_spinner=False)
def get_pipeline_cfg() -> dict:
    return load_config()["pipeline"]


@st.cache_data(show_spinner=False)
def get_gsd_cfg() -> dict:
    return load_config()["gsd"]


GSD_AUTO = "Auto (GeoTIFF)"
GSD_ALTITUDE = "Flight altitude"
GSD_PIXEL = "Pixel size"


PREVIEW_MAX_SIDE = 2000   # on-page preview
EXPORT_MAX_SIDE = 12000   # full-resolution PNG; caps RAM (~430 MB) for huge orthomosaics


def read_rgb(path: str, max_side: int, resampling: Resampling) -> tuple[np.ndarray, float]:
    """Whole image as uint8 RGB, downscaled so its long side is <= max_side."""
    with rasterio.open(path) as src:
        h, w = src.height, src.width
        scale = min(1.0, max_side / max(h, w))
        out_h, out_w = max(1, round(h * scale)), max(1, round(w * scale))
        n_bands = min(3, src.count)
        indexes = list(range(1, n_bands + 1))  # rasterio bands are 1-indexed
        if scale == 1.0:
            data = src.read(indexes=indexes)
        else:
            data = src.read(
                indexes=indexes, out_shape=(n_bands, out_h, out_w), resampling=resampling,
            )
    if n_bands < 3:
        data = np.repeat(data[:1], 3, axis=0)
    rgb = np.moveaxis(data[:3], 0, -1)
    if rgb.dtype != np.uint8:
        mx = (
            np.iinfo(rgb.dtype).max
            if np.issubdtype(rgb.dtype, np.integer)
            else float(rgb.max()) or 1.0
        )
        rgb = (rgb.astype(np.float32) / mx * 255).clip(0, 255).astype(np.uint8)
    # OpenCV draws only on contiguous arrays
    return np.ascontiguousarray(rgb), scale


def remove_file(path: str | None) -> None:
    if path and os.path.exists(path):
        try:
            os.unlink(path)
        except OSError:
            pass  # Windows may still hold the file; it lives in the temp folder anyway


def clear_results() -> None:
    """Drop results and the uploaded copy kept on disk for the full-size export."""
    if st.session_state.get("results"):
        remove_file(st.session_state.results.get("source_path"))
    st.session_state.results = None


class ResolutionError(Exception):
    """The image's pixel size is unusable for detection (shown to the user as is)."""


def resolve_gsd(job: dict, path: str) -> tuple[float | None, str]:
    """GSD of the job's image in cm/px (None if unknown) and a note on its source."""
    mode = job.get("gsd_mode", GSD_AUTO)
    if mode == GSD_ALTITUDE:
        cam = get_gsd_cfg()["camera"]
        gsd = gsd_from_altitude(job["gsd_value"], **cam)
        return gsd, f"from flight altitude {job['gsd_value']:g} m"
    if mode == GSD_PIXEL:
        return job["gsd_value"], "entered manually"
    gsd = gsd_from_metadata(path)
    if gsd is None:
        return None, "no georeferencing found, image processed at original scale"
    return gsd, "from GeoTIFF metadata"


# Page setup
st.set_page_config(layout="wide", page_title="Penguin Detector", page_icon="🐧")
apply_custom_styles()
render_hero()

st.markdown(f"""
<div class="fact-card">
    <span class="fact-icon">🐧</span>
    <p class="fact-text"><strong>Did you know?</strong> {random.choice(PENGUIN_FACTS)}</p>
</div>
""", unsafe_allow_html=True)

for key, default in (("results", None), ("job", None), ("error", None), ("upload_id", None)):
    if key not in st.session_state:
        st.session_state[key] = default

# While a detection is running, every widget is disabled: in Streamlit any widget
# change reruns the script, which would silently abort a long detection.
busy = st.session_state.job is not None

# Sidebar
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    st.divider()

    model_options = list_models()
    model_name = st.selectbox(
        "Model",
        options=list(model_options.keys()),
        format_func=lambda k: model_options[k],
        disabled=busy,
    )
    conf = st.slider("Confidence threshold", 0.10, 1.0, 0.25, 0.05, disabled=busy)

    st.divider()
    gsd_mode = st.radio(
        "Image resolution",
        options=[GSD_AUTO, GSD_ALTITUDE, GSD_PIXEL],
        disabled=busy,
        help=(
            "The models were trained on photos taken from 80–100 m. Images taken "
            "from higher up are rescaled so penguins appear at the trained size. "
            "Auto reads the pixel size from a GeoTIFF; for JPEG/PNG enter the "
            "flight altitude (raw drone photos) or the pixel size the orthomosaic "
            "was exported with (Metashape: Export Orthomosaic → Pixel size)."
        ),
    )
    gsd_value = None
    if gsd_mode == GSD_ALTITUDE:
        gsd_value = st.number_input(
            "Flight altitude, m", min_value=10.0, max_value=500.0,
            value=100.0, step=5.0, disabled=busy,
        )
    elif gsd_mode == GSD_PIXEL:
        gsd_value = st.number_input(
            "Pixel size, cm/px", min_value=0.1, max_value=50.0,
            value=2.7, step=0.1, format="%.2f", disabled=busy,
        )

# Upload
uploaded = st.file_uploader(
    "Aerial image",
    type=["jpg", "jpeg", "png", "tif", "tiff"],
    disabled=busy,
)

# A new image makes the previous results meaningless
if uploaded is not None and not busy:
    upload_id = (uploaded.name, uploaded.size)
    if upload_id != st.session_state.upload_id:
        st.session_state.upload_id = upload_id
        clear_results()
        st.session_state.error = None

if uploaded and not busy:
    if gsd_mode == GSD_AUTO and not uploaded.name.lower().endswith((".tif", ".tiff")):
        st.info(
            "JPEG/PNG files carry no pixel size, so the image will be processed at "
            "its original scale. If it was taken from above 100 m, set "
            "**Image resolution** in the sidebar to the flight altitude or pixel size."
        )
    st.divider()
    if st.button("🔍 Detect Penguins", type="primary", use_container_width=True):
        clear_results()  # frees the previous image's copy on disk first
        # Save the upload to disk first, so the job no longer depends on the widget
        ext = os.path.splitext(uploaded.name)[1] or ".tif"
        with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as f:
            uploaded.seek(0)
            shutil.copyfileobj(uploaded, f, length=16 * 1024 * 1024)
        st.session_state.job = {
            "path": f.name,
            "name": uploaded.name,
            "model": model_name,
            "conf": conf,
            "gsd_mode": gsd_mode,
            "gsd_value": gsd_value,
        }
        st.session_state.error = None
        st.rerun()

if busy:
    job = st.session_state.job
    tmp_src = job["path"]
    cfg = get_pipeline_cfg()
    st.divider()
    st.info("Detection is running. Please keep this page open until it finishes.")
    progress = st.progress(0, text="Initialising…")

    try:
        progress.progress(5, text="Loading model…")
        model = get_model(job["model"])

        progress.progress(10, text="Reading image…")
        with rasterio.open(tmp_src) as src:
            h, w = src.height, src.width

        gsd_cfg = get_gsd_cfg()
        gsd_cm, gsd_note = resolve_gsd(job, tmp_src)
        # Upsample coarse images (e.g. 3.3 cm/px from 120 m) to the training GSD
        tile_scale = gsd_cm / gsd_cfg["reference_cm"] if gsd_cm else 1.0
        if not gsd_cfg["min_scale"] <= tile_scale <= gsd_cfg["max_scale"]:
            raise ResolutionError(
                f"Pixel size {gsd_cm:.2f} cm/px ({gsd_note}) is too far from the "
                f"training resolution {gsd_cfg['reference_cm']} cm/px. "
                "Check the Image resolution setting."
            )

        t_start = time.perf_counter()

        def on_tile(done: int, total: int) -> None:
            pct = 10 + int(done / total * 75)
            elapsed = time.perf_counter() - t_start
            left = elapsed / done * (total - done)
            eta = f"~{left / 60:.0f} min left" if left >= 90 else f"~{left:.0f} s left"
            progress.progress(pct, text=f"Tile {done} / {total} · {eta}")

        raw_boxes, raw_scores = run_inference(
            tmp_src, model, job["conf"],
            cfg["tile_size"], cfg["overlap"],
            on_tile=on_tile,
            batch_size=cfg.get("batch_size", 4),
            scale=tile_scale,
        )

        progress.progress(88, text="Aggregating predictions (NMS)…")
        boxes, scores = apply_nms(raw_boxes, raw_scores, cfg["iou_threshold"])
        elapsed_sec = time.perf_counter() - t_start

        progress.progress(95, text="Drawing results…")
        original_display, scale = read_rgb(tmp_src, PREVIEW_MAX_SIDE, Resampling.lanczos)

        progress.progress(100, text="Done!")

        st.session_state.results = {
            "source_path": tmp_src,  # kept for the full-resolution PNG export
            "full_png": None,        # (confidence, png bytes) once prepared
            "original": original_display,
            "boxes": boxes,
            "scores": scores,
            "scale": scale,
            "image_shape": (h, w),
            "image_name": job["name"],
            "elapsed_sec": elapsed_sec,
            "inference_conf": job["conf"],
            "gsd_cm": gsd_cm,
            "gsd_note": gsd_note,
            "tile_scale": tile_scale,
        }

    except ResolutionError as e:
        st.session_state.error = (str(e), None)
    except MemoryError:
        st.session_state.error = (
            "Not enough memory (RAM) to process this image. "
            "Close other programs and try again.",
            None,
        )
    except rasterio.errors.RasterioIOError as e:
        st.session_state.error = (
            "Could not read this image. The file may be damaged or in an "
            "unsupported format. Supported: JPG, PNG, TIFF.",
            str(e),
        )
    except Exception:
        st.session_state.error = (
            "Something went wrong during detection. Please try again; if it "
            "keeps happening, send the details below to the developer.",
            traceback.format_exc(),
        )
    finally:
        st.session_state.job = None
        # On success the copy is kept for the full-resolution export (removed by
        # clear_results on the next detection or upload)
        if not st.session_state.results:
            remove_file(tmp_src)

    # Rerun so the widgets are enabled again and results/errors are shown
    st.rerun()

if st.session_state.error:
    message, details = st.session_state.error
    st.error(message)
    if details:
        with st.expander("Technical details"):
            st.code(details, language=None)

# Results
if st.session_state.results:
    r = st.session_state.results

    # Re-filter stored detections by current slider value so the slider
    # works interactively without re-running inference.
    if conf < r["inference_conf"]:
        st.info(
            f"Confidence {conf:.0%} is below the inference threshold "
            f"{r['inference_conf']:.0%}. Re-run detection to see more results."
        )

    filtered = [(b, s) for b, s in zip(r["boxes"], r["scores"]) if s >= conf]
    # Number penguins in reading order (rows of one bird's height, top to bottom,
    # left to right within a row) so neighbouring numbers sit next to each other.
    # The same order is used for the picture, RoiSet.zip and Label Studio JSON.
    if filtered:
        row_h = max(1.0, float(np.median([b[3] - b[1] for b, _ in filtered])))
        filtered.sort(key=lambda bs: (
            int((bs[0][1] + bs[0][3]) / 2 // row_h), (bs[0][0] + bs[0][2]) / 2,
        ))
    f_boxes = [b for b, _ in filtered]
    f_scores = [s for _, s in filtered]

    m = compute_metrics(f_boxes, f_scores)
    total = m["total"]

    scale = r["scale"]
    display_boxes = [
        (int(x1 * scale), int(y1 * scale), int(x2 * scale), int(y2 * scale))
        for x1, y1, x2, y2 in f_boxes
    ]
    numbers = [str(i) for i in range(1, len(display_boxes) + 1)]
    result_display = draw_boxes(r["original"], display_boxes, numbers)

    st.divider()

    if total == 0:
        st.warning("No penguins detected — they might be hiding in the snow!")
        badge = "🥶 Snow Scout"
    elif total < 5:
        st.success(f"Small colony detected: **{total}** penguins.")
        badge = "🐧 Colony Finder"
    elif total < 20:
        st.success(f"Nice find! Detected **{total}** penguins.")
        badge = "🏔️ Polar Explorer"
    else:
        st.success(f"Penguin paradise! Detected **{total}** penguins.")
        badge = "👑 Penguin Master"

    st.markdown(f'<div class="badge-pill">{badge}</div>', unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Penguins detected", total)
    col2.metric("Avg confidence", f"{m['avg_confidence']:.1%}")
    col3.metric("Max confidence", f"{m['max_confidence']:.1%}")
    elapsed = r["elapsed_sec"]
    col4.metric(
        "Processing time",
        f"{int(elapsed) // 60}m {int(elapsed) % 60}s" if elapsed >= 60 else f"{elapsed:.1f} s",
    )

    if r["gsd_cm"]:
        st.caption(
            f"Pixel size {r['gsd_cm']:.2f} cm/px ({r['gsd_note']}); "
            f"image rescaled ×{r['tile_scale']:.2f} to the training resolution."
        )
    else:
        st.caption(f"Pixel size unknown: {r['gsd_note']}.")

    st.divider()

    img_col1, img_col2 = st.columns(2)
    with img_col1:
        st.image(r["original"], caption="Original image", use_container_width=True)
    with img_col2:
        st.image(result_display, caption="Detected penguins", use_container_width=True)

    st.divider()

    # Export
    st.markdown('<p class="section-label">Export results</p>', unsafe_allow_html=True)
    dl1, dl2, dl3 = st.columns(3)

    with dl1:
        st.download_button(
            "⬇️ RoiSet.zip — ImageJ",
            to_roi_zip(f_boxes),
            "RoiSet.zip",
            mime="application/zip",
            use_container_width=True,
            help="Open in Fiji / ImageJ: Analyze → Tools → ROI Manager → More → Open",
        )
    with dl2:
        img_h, img_w = r["image_shape"]
        exp_scale = min(1.0, EXPORT_MAX_SIDE / max(img_h, img_w))
        exp_size = f"{round(img_w * exp_scale)}×{round(img_h * exp_scale)}"
        # Full-size PNG is heavy (seconds to minutes, up to ~430 MB of RAM), so it is
        # drawn only on request and kept for the current confidence threshold.
        prepared = r["full_png"]
        if prepared and prepared[0] == conf:
            st.download_button(
                f"⬇️ Image (PNG, {exp_size})",
                prepared[1],
                "penguin_detection.png",
                mime="image/png",
                use_container_width=True,
                help="Full-resolution image with numbered boxes: zoom in to check each penguin.",
            )
        elif st.button(
            f"🖼️ Prepare image (PNG, {exp_size})",
            use_container_width=True,
            help="Draws the numbered boxes on the full-resolution image. "
                 "Large orthomosaics take up to a few minutes.",
        ):
            r["full_png"] = None  # free the previous image before drawing a new one
            try:
                with st.spinner("Drawing the full-resolution image…"):
                    full, s = read_rgb(r["source_path"], EXPORT_MAX_SIDE, Resampling.average)
                    full_boxes = [
                        (round(x1 * s), round(y1 * s), round(x2 * s), round(y2 * s))
                        for x1, y1, x2, y2 in f_boxes
                    ]
                    draw_boxes(full, full_boxes, numbers, inplace=True)
                    png = to_bytes(full, compress_level=3)
                    del full
                r["full_png"] = (conf, png)
                st.rerun()
            except MemoryError:
                st.error("Not enough memory (RAM) for the full-resolution image. "
                         "Close other programs and try again.")
            except rasterio.errors.RasterioIOError:
                st.error("The original image is no longer available. "
                         "Run Detect Penguins again.")
    with dl3:
        st.download_button(
            "⬇️ Label Studio JSON",
            to_label_studio_json(f_boxes, f_scores, img_w, img_h, r["image_name"]),
            "penguin_detections.json",
            mime="application/json",
            use_container_width=True,
            help="Import in Label Studio: Tasks → Import → JSON",
        )
