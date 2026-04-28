import streamlit as st


def apply_custom_styles():
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

        html, body, [class*="css"], * {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
        }

        #MainMenu, footer, header { visibility: hidden; }

        /* ── Page background ── */
        .stApp { background-color: #f0f7fc; }

        /* ── Hero ── */
        .hero-wrap {
            padding: 2.5rem 3rem;
            border-radius: 24px;
            background: linear-gradient(135deg, #0096c7 0%, #005f8a 100%);
            box-shadow: 0 8px 32px rgba(0, 100, 180, 0.22);
            margin-bottom: 1.75rem;
            position: relative;
            overflow: hidden;
        }
        .hero-wrap::after {
            content: '';
            position: absolute;
            bottom: -60px; right: -30px;
            width: 220px; height: 220px;
            background: radial-gradient(circle, rgba(255,255,255,0.10) 0%, transparent 70%);
            pointer-events: none;
        }
        .hero-tag {
            display: inline-block;
            background: rgba(255,255,255,0.18);
            border: 1px solid rgba(255,255,255,0.30);
            color: rgba(255,255,255,0.92);
            padding: 0.18rem 0.8rem;
            border-radius: 20px;
            font-size: 0.67rem;
            font-weight: 600;
            letter-spacing: 1.6px;
            text-transform: uppercase;
            margin-bottom: 0.9rem;
        }
        .hero-title {
            font-size: 2.1rem;
            font-weight: 700;
            color: #ffffff;
            margin: 0 0 0.45rem 0;
            letter-spacing: -0.4px;
            line-height: 1.15;
        }
        .hero-sub {
            font-size: 0.92rem;
            color: rgba(255,255,255,0.78);
            margin: 0;
            max-width: 520px;
            line-height: 1.55;
        }

        /* ── Fact card ── */
        .fact-card {
            display: flex;
            align-items: flex-start;
            gap: 0.75rem;
            background: #ffffff;
            border: 1px solid rgba(0, 150, 199, 0.18);
            border-left: 4px solid #0096c7;
            border-radius: 12px;
            padding: 0.9rem 1.2rem;
            margin-bottom: 1.5rem;
            box-shadow: 0 2px 8px rgba(0, 100, 180, 0.06);
        }
        .fact-icon { font-size: 1.3rem; line-height: 1.4; }
        .fact-text {
            font-size: 0.88rem;
            color: #2a5070;
            line-height: 1.55;
            margin: 0;
        }

        /* ── Badge ── */
        .badge-pill {
            display: inline-flex;
            align-items: center;
            gap: 0.45rem;
            padding: 0.5rem 1.25rem;
            border-radius: 50px;
            background: linear-gradient(135deg, #e3f4fb 0%, #c8eaf7 100%);
            border: 1.5px solid rgba(0, 150, 199, 0.30);
            color: #005f8a;
            font-size: 1rem;
            font-weight: 600;
            margin: 0.1rem 0 1rem 0;
        }

        /* ── Section label ── */
        .section-label {
            font-size: 0.65rem;
            font-weight: 700;
            letter-spacing: 2px;
            text-transform: uppercase;
            color: #7aaec8;
            margin: 0 0 0.5rem 0;
        }

        /* ── Metrics ── */
        [data-testid="metric-container"] {
            background: #ffffff !important;
            border: 1px solid rgba(0, 150, 199, 0.16) !important;
            border-radius: 14px !important;
            padding: 1rem 1.2rem !important;
            box-shadow: 0 2px 10px rgba(0, 100, 180, 0.06) !important;
        }
        [data-testid="stMetricValue"] {
            color: #0096c7 !important;
            font-weight: 700 !important;
        }
        [data-testid="stMetricLabel"] { color: #5a8ab0 !important; }

        /* ── Sidebar ── */
        section[data-testid="stSidebar"] {
            background: #ddeef8 !important;
            border-right: 1px solid rgba(0, 150, 199, 0.12) !important;
        }

        /* ── Primary button ── */
        .stButton > button[kind="primary"] {
            background: #0096c7 !important;
            border-color: #0096c7 !important;
            color: #ffffff !important;
        }
        .stButton > button[kind="primary"]:hover {
            background: #007aab !important;
            border-color: #007aab !important;
        }
        .stButton > button[kind="primary"]:active {
            background: #005f8a !important;
            border-color: #005f8a !important;
        }

        /* ── Divider ── */
        hr {
            border: none !important;
            border-top: 1px solid rgba(0, 150, 199, 0.13) !important;
            margin: 1.25rem 0 !important;
        }
    </style>
    """, unsafe_allow_html=True)
