"""
deploy/app.py — Streamlit dashboard for HuggingFace Spaces deployment.

Features:
  - Live forecast (1d / 2d / 3d / 7d)
  - BUY / WAIT / STOCK UP / LIQUIDATE / HOLD signal
  - VaR display (worst-case ₹ loss)
  - Regime indicator (LOW / HIGH volatility)
  - MCX price chart
  - Daily auto-refresh

CPU-only inference — no GPU required.
"""
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta

# ── Dependencies import guard ──────────────────────────────────────────────
try:
    from utils.config_loader import load_config
    from data.ingest import ingest_all, save_raw
    from data.preprocess import preprocess, save_processed
    from features.engineer import build_features, save_features
    from regime.hmm_model import run_hmm_pipeline
    from regime.garch_model import add_garch_features
    from inference.predictor import predict_all_horizons
    from inference.decision_engine import generate_signal, weekly_calibrate, load_threshold
except ImportError as e:
    st.error(f"Import error: {e}. Ensure all modules are on PYTHONPATH.")
    st.stop()

# ── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="🥇 Gold Forecast — Indian Market",
    page_icon="🥇",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Signal Colour Mapping ────────────────────────────────────────────────────
SIGNAL_COLORS = {
    "BUY NOW":           "#00c853",
    "STOCK UP":          "#69f0ae",
    "HOLD INVENTORY":    "#ffd54f",
    "WAIT":              "#ff7043",
    "LIQUIDATE PARTIAL": "#d50000",
}

# ── Data Loading (cached) ────────────────────────────────────────────────────
@st.cache_data(ttl=3600, show_spinner="Fetching latest market data...")
def get_fresh_data():
    cfg = load_config()
    # Ingest
    raw = ingest_all(cfg)
    save_raw(raw, cfg)
    # Preprocess
    proc = preprocess(raw, cfg)
    save_processed(proc, cfg)
    # Features
    feat = build_features(proc, cfg)
    # Regime
    feat = run_hmm_pipeline(feat, cfg)
    feat = add_garch_features(feat, cfg)
    save_features(feat, cfg)
    return feat, cfg


@st.cache_data(ttl=3600, show_spinner="Running model inference...")
def get_predictions(df_hash):
    cfg = load_config()
    feat = pd.read_parquet(os.path.join(cfg["paths"]["processed_data"], "features_daily.parquet"))
    preds = predict_all_horizons(feat, cfg)
    weekly_calibrate(feat, cfg)
    signal = generate_signal(preds, feat, cfg)
    return preds, signal


# ── Sidebar ──────────────────────────────────────────────────────────────────
st.sidebar.title("🥇 Gold Forecasting")
st.sidebar.markdown("**Indian Jewellery Market Intelligence**")
st.sidebar.markdown("---")
st.sidebar.markdown("**System Specs**")
st.sidebar.markdown("- Model: XGBoost per regime×horizon")
st.sidebar.markdown("- Regime: 2-state HMM")
st.sidebar.markdown("- Risk: VaR (5th percentile)")
st.sidebar.markdown("- Target: MCX ₹/10g")
st.sidebar.markdown("---")
refresh = st.sidebar.button("🔄 Refresh Now")

# ── Main ─────────────────────────────────────────────────────────────────────
st.title("🥇 Gold Price Forecast — Indian Jewellery Market")
st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M IST')}")

# Load data
try:
    df, cfg = get_fresh_data()
    df_hash = str(df.shape) + str(df.index[-1])
    preds, signal = get_predictions(df_hash)
except FileNotFoundError:
    st.error("⚠️ Models not trained yet. Run `python train/trainer.py` to train models first.")
    st.info("Once trained, restart this app.")
    st.stop()
except Exception as e:
    st.error(f"Pipeline error: {e}")
    st.stop()

# ── Signal Card ──────────────────────────────────────────────────────────────
sig_name = signal.get("signal", "HOLD INVENTORY")
sig_color = SIGNAL_COLORS.get(sig_name, "#78909c")

st.markdown("---")
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(
        f"<div style='background:{sig_color};padding:24px;border-radius:12px;text-align:center'>"
        f"<h2 style='color:white;margin:0'>{sig_name}</h2></div>",
        unsafe_allow_html=True,
    )

with col2:
    st.metric("MCX Price (₹/10g)", f"₹{signal.get('current_mcx_approx', 0):,.0f}")
    st.metric("1-Day Expected Move", f"₹{signal.get('pred_1d_inr', 0):+,.0f}")

with col3:
    st.metric("7-Day Expected (%)", f"{signal.get('pred_7d_pct', 0):+.2f}%")
    st.metric("VaR Worst Case", f"₹{signal.get('var_inr', 0):,.0f}")

with col4:
    st.metric("Regime", signal.get("regime_label", "UNKNOWN"))
    st.metric("Confidence Band ±", f"₹{signal.get('confidence_band_inr', 0):,.0f}")

st.info(f"**Reason:** {signal.get('reason', '')}")

# ── Forecast Table ───────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("📊 Forecast by Horizon")

rows = []
for h, p in preds.items():
    if "error" not in p:
        rows.append({
            "Horizon": f"{h} Day",
            "Expected ₹ Change": f"₹{p['predicted_inr_change']:+,.0f}",
            "Expected MCX Price": f"₹{p['predicted_inr_price']:,.0f}",
            "Direction Confidence": f"{p['direction_prob']:.1%}",
            "Regime": p["regime_label"],
        })

if rows:
    st.dataframe(pd.DataFrame(rows), use_container_width=True)

# ── MCX Price Chart ───────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("📈 MCX Gold Price — Historical (₹/10g)")

chart_df = df[["mcx_approx", "regime"]].dropna().tail(365)

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=chart_df.index, y=chart_df["mcx_approx"],
    mode="lines", name="MCX ₹/10g",
    line=dict(color="#FFD700", width=2)
))

# Shade HIGH regime periods
high_mask = chart_df["regime"] == 1
if high_mask.any():
    transitions = high_mask.astype(int).diff().fillna(0)
    starts = chart_df.index[transitions == 1].tolist()
    ends = chart_df.index[transitions == -1].tolist()
    if high_mask.iloc[0]:
        starts = [chart_df.index[0]] + starts
    if high_mask.iloc[-1]:
        ends.append(chart_df.index[-1])
    for s, e in zip(starts, ends):
        fig.add_vrect(x0=s, x1=e, fillcolor="red", opacity=0.12, line_width=0,
                     annotation_text="HIGH" if (e - s).days > 10 else "")

fig.update_layout(
    template="plotly_dark",
    paper_bgcolor="#0e1117",
    plot_bgcolor="#0e1117",
    hovermode="x unified",
    margin=dict(l=20, r=20, t=20, b=20),
    legend=dict(orientation="h"),
    xaxis_title="Date",
    yaxis_title="₹ per 10g",
)
st.plotly_chart(fig, use_container_width=True)

# ── Disclaimer ───────────────────────────────────────────────────────────────
st.markdown("---")
st.caption(
    "⚠️ **Disclaimer**: This is a forecasting model for decision support only. "
    "Gold markets are inherently unpredictable. Never rely solely on AI predictions. "
    "Combine with your own market knowledge, cash flow needs, and inventory requirements."
)
