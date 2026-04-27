import os
import random
import tempfile
import time

import numpy as np
import rasterio
import streamlit as st
from rasterio.enums import Resampling

from core.components import render_hero
from core.config import PENGUIN_FACTS
from core.styles import apply_custom_styles
from models.model_registry import list_models, load, load_config
from pipeline.counter import compute_metrics
from pipeline.inference import run as run_inference
from pipeline.nms import apply_nms
from utils.exporter import to_label_studio_json, to_roi_zip
from utils.visualizer import draw_boxes, to_bytes


@st.cache_resource
def get_model(name: str):
    return load(name)


@st.cache_data
def get_pipeline_cfg() -> dict:
    return load_config()["pipeline"]


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

# Sidebar
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    st.divider()

    model_options = list_models()
    model_name = st.selectbox(
        "Model",
        options=list(model_options.keys()),
        format_func=lambda k: model_options[k],
    )
    conf = st.slider("Confidence threshold", 0.10, 1.0, 0.25, 0.05)

# Upload
uploaded = st.file_uploader(
    "Aerial image",
    type=["jpg", "jpeg", "png", "tif", "tiff"],
)

if "results" not in st.session_state:
    st.session_state.results = None

if uploaded:
    st.divider()
    if st.button("🔍 Detect Penguins", type="primary", use_container_width=True):
        st.session_state.results = None
        cfg = get_pipeline_cfg()
        progress = st.progress(0, text="Initialising…")
        tmp_src = None

        try:
            progress.progress(5, text="Loading model…")
            model = get_model(model_name)

            progress.progress(10, text="Reading image…")
            ext = os.path.splitext(uploaded.name)[1] or ".tif"
            with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as f:
                f.write(uploaded.read())
                tmp_src = f.name

            with rasterio.open(tmp_src) as src:
                h, w = src.height, src.width

            mpx = w * h / 1_000_000
            if mpx > 500:
                st.info(
                    f"Large image: {w} × {h} px ({mpx:.0f} MP). "
                    "Tiles are read from disk — RAM usage stays low."
                )

            def on_tile(done: int, total: int) -> None:
                pct = 10 + int(done / total * 75)
                progress.progress(pct, text=f"Tile {done} / {total}")

            t_start = time.perf_counter()
            raw_boxes, raw_scores = run_inference(
                tmp_src, model, conf,
                cfg["tile_size"], cfg["overlap"],
                on_tile=on_tile,
                batch_size=cfg.get("batch_size", 4),
            )

            progress.progress(88, text="Aggregating predictions (NMS)…")
            boxes, scores = apply_nms(raw_boxes, raw_scores, cfg["iou_threshold"])
            elapsed_sec = time.perf_counter() - t_start

            progress.progress(95, text="Drawing results…")
            scale = min(1.0, 2000 / max(h, w))
            out_h = max(1, int(h * scale))
            out_w = max(1, int(w * scale))
            with rasterio.open(tmp_src) as src:
                n_bands = min(3, src.count)
                indexes = list(range(1, n_bands + 1))  # rasterio bands are 1-indexed
                data = src.read(
                    indexes=indexes,
                    out_shape=(n_bands, out_h, out_w),
                    resampling=Resampling.lanczos,
                )
                if n_bands < 3:
                    data = np.repeat(data[:1], 3, axis=0)
                original_display = np.moveaxis(data[:3], 0, -1)
                if original_display.dtype != np.uint8:
                    mx = (
                        np.iinfo(original_display.dtype).max
                        if np.issubdtype(original_display.dtype, np.integer)
                        else float(original_display.max()) or 1.0
                    )
                    original_display = (
                        original_display.astype(np.float32) / mx * 255
                    ).clip(0, 255).astype(np.uint8)

            display_boxes = [
                (int(x1 * scale), int(y1 * scale), int(x2 * scale), int(y2 * scale))
                for x1, y1, x2, y2 in boxes
            ]
            result_display = draw_boxes(original_display, display_boxes, scores)

            progress.progress(100, text="Done!")

            st.session_state.results = {
                "metrics": compute_metrics(boxes, scores),
                "original": original_display,
                "result_rgb": result_display,
                "boxes": boxes,
                "scores": scores,
                "image_shape": (h, w),
                "image_name": uploaded.name,
                "elapsed_sec": elapsed_sec,
            }

        except FileNotFoundError as e:
            progress.empty()
            st.error(str(e))
        except MemoryError:
            progress.empty()
            st.error(
                "Not enough RAM. Try increasing the Docker memory limit."
            )
        finally:
            if tmp_src and os.path.exists(tmp_src):
                os.unlink(tmp_src)

# Results
if st.session_state.results:
    r = st.session_state.results
    m = r["metrics"]
    total = m["total"]

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

    st.divider()

    img_col1, img_col2 = st.columns(2)
    with img_col1:
        st.image(r["original"], caption="Original image", use_container_width=True)
    with img_col2:
        st.image(r["result_rgb"], caption="Detected penguins", use_container_width=True)

    st.divider()

    # Export
    st.markdown('<p class="section-label">Export results</p>', unsafe_allow_html=True)
    dl1, dl2, dl3 = st.columns(3)

    with dl1:
        st.download_button(
            "⬇️ RoiSet.zip — ImageJ",
            to_roi_zip(r["boxes"]),
            "RoiSet.zip",
            mime="application/zip",
            use_container_width=True,
            help="Open in Fiji / ImageJ: Analyze → Tools → ROI Manager → More → Open",
        )
    with dl2:
        st.download_button(
            "⬇️ Image (PNG)",
            to_bytes(r["result_rgb"]),
            "penguin_detection.png",
            mime="image/png",
            use_container_width=True,
        )
    with dl3:
        img_h, img_w = r["image_shape"]
        st.download_button(
            "⬇️ Label Studio JSON",
            to_label_studio_json(r["boxes"], r["scores"], img_w, img_h, r["image_name"]),
            "penguin_detections.json",
            mime="application/json",
            use_container_width=True,
            help="Import in Label Studio: Tasks → Import → JSON",
        )
