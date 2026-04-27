import streamlit as st


def render_hero():
    st.markdown("""
    <div class="hero-wrap">
        <div class="hero-tag">YOLOv11 · RF-DETR</div>
        <h1 class="hero-title">🐧 Penguin Detector</h1>
        <p class="hero-sub">
            Upload an aerial image and the model will tile, scan, and count
            every penguin in the colony automatically.
        </p>
    </div>
    """, unsafe_allow_html=True)


def render_fact(fact: str):
    st.markdown(f"""
    <div class="fact-card">
        <span class="fact-icon">🐧</span>
        <p class="fact-text"><strong>Did you know?</strong> {fact}</p>
    </div>
    """, unsafe_allow_html=True)


def render_badge(badge: str):
    st.markdown(f'<div class="badge-pill">{badge}</div>', unsafe_allow_html=True)
