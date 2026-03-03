"""
app.py — Root-level Gradio interface for HuggingFace Spaces (CPU-only).

Architecture:
  - On launch: ingests fresh data, builds features, detects regime, trains XGBoost models
  - Subsequent runs use cached artifacts if they exist
  - No GPU required: XGBoost + HMM is CPU-native
  - Full error handling with graceful degradation

HuggingFace Space: ShivamBharti085/gold-forecast-india
"""
import os
import sys
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import gradio as gr
import pandas as pd
import numpy as np
from datetime import datetime

from utils.config_loader import load_config
from utils.logger import get_logger

log = get_logger("app")

SIGNAL_LABELS = {
    "BUY NOW":           "GREEN  |  BUY NOW",
    "STOCK UP":          "LIME   |  STOCK UP",
    "HOLD INVENTORY":    "YELLOW |  HOLD INVENTORY",
    "WAIT":              "ORANGE |  WAIT",
    "LIQUIDATE PARTIAL": "RED    |  LIQUIDATE PARTIAL",
}

_pipeline_cache = {"df": None, "cfg": None, "ready": False}


def run_pipeline_once(cfg):
    """Run full pipeline: ingest -> preprocess -> features -> regime -> train."""
    log.info("Running pipeline (first-time or refresh)...")

    from data.ingest import ingest_all, save_raw
    from data.preprocess import preprocess, save_processed
    from features.engineer import build_features, save_features
    from regime.hmm_model import run_hmm_pipeline
    from regime.garch_model import add_garch_features
    from train.trainer import train_all_models

    raw = ingest_all(cfg)
    save_raw(raw, cfg)

    proc = preprocess(raw, cfg)
    save_processed(proc, cfg)

    feat = build_features(proc, cfg)
    feat = run_hmm_pipeline(feat, cfg)
    feat = add_garch_features(feat, cfg)
    save_features(feat, cfg)

    train_all_models(feat, cfg)
    return feat


def get_or_build_data(cfg):
    """Load cached data or rebuild."""
    if _pipeline_cache["ready"] and _pipeline_cache["df"] is not None:
        return _pipeline_cache["df"]

    feat_path = os.path.join(cfg["paths"]["processed_data"], "features_daily.parquet")
    model_path = os.path.join(cfg["paths"]["model_artifacts"], "feature_cols.pkl")

    if os.path.exists(feat_path) and os.path.exists(model_path):
        log.info("Loading cached artifacts from disk...")
        df = pd.read_parquet(feat_path)
    else:
        df = run_pipeline_once(cfg)

    _pipeline_cache["df"] = df
    _pipeline_cache["ready"] = True
    return df


def get_forecast():
    """Main forecast function called by Gradio button."""
    try:
        cfg = load_config()

        # Ensure output directories exist
        for k, p in cfg["paths"].items():
            if not p.endswith(".json"):
                os.makedirs(p, exist_ok=True)

        df = get_or_build_data(cfg)

        from inference.predictor import predict_all_horizons
        from inference.decision_engine import generate_signal, weekly_calibrate

        weekly_calibrate(df, cfg)
        preds = predict_all_horizons(df, cfg)
        signal = generate_signal(preds, df, cfg)

        sig_name = signal.get("signal", "HOLD INVENTORY")
        sig_display = SIGNAL_LABELS.get(sig_name, sig_name)
        mcx = signal.get("current_mcx_approx", 0)
        move_1d = signal.get("pred_1d_inr", 0)
        move_7d = signal.get("pred_7d_pct", 0)
        var_inr = signal.get("var_inr", 0)
        band = signal.get("confidence_band_inr", 0)
        reason = signal.get("reason", "")
        regime = signal.get("regime_label", "UNKNOWN")
        threshold = signal.get("threshold_used", 0)
        ts = signal.get("timestamp", datetime.now().strftime("%Y-%m-%d %H:%M"))

        rows = []
        for h in [1, 2, 3, 7]:
            p = preds.get(h, {})
            if "error" not in p:
                rows.append({
                    "Horizon": f"{h}d",
                    "Expected Rs. Change": f"Rs.{p.get('predicted_inr_change', 0):+,.0f}",
                    "Expected MCX Price": f"Rs.{p.get('predicted_inr_price', mcx):,.0f}",
                    "Direction Confidence": f"{p.get('direction_prob', 0.5):.1%}",
                    "Regime": p.get("regime_label", "?"),
                })

        table_df = pd.DataFrame(rows) if rows else pd.DataFrame({"Status": ["No predictions"]})

        return (
            sig_display,
            f"Rs.{mcx:,.0f}",
            f"Rs.{move_1d:+,.0f}",
            f"{move_7d:+.2f}%",
            f"Rs.{var_inr:,.0f}",
            f"Rs.{band:,.0f}",
            regime,
            f"{threshold:.3f}",
            reason,
            ts,
            table_df,
        )

    except Exception as e:
        import traceback
        err = f"Error: {e}\n\nTrace:\n{traceback.format_exc()[:1000]}"
        log.error(err)
        empty_df = pd.DataFrame({"Status": [str(e)]})
        return (f"ERROR: {e}", "-", "-", "-", "-", "-", "-", "-", str(e), "-", empty_df)


# ── Gradio UI ─────────────────────────────────────────────────────────────────
with gr.Blocks(
    title="Gold Forecast | Indian Jewellery Market",
    theme=gr.themes.Soft(primary_hue="orange"),
) as demo:

    gr.Markdown("""
    # Gold Price Forecast — Indian Jewellery Market
    **Production ML System** | MCX Gold (Rs./10g) | 1d / 2d / 3d / 7d Horizons
    Model: XGBoost per Regime | Regime: 2-state HMM + GARCH | Risk: VaR(5th pct)

    > First run takes ~90s (data fetch + model training on CPU). Subsequent runs are fast.
    """)

    refresh_btn = gr.Button("Run Forecast", variant="primary")

    gr.Markdown("---")
    with gr.Row():
        signal_box  = gr.Textbox(label="SIGNAL", scale=2)
        ts_box      = gr.Textbox(label="Updated At", scale=1)

    with gr.Row():
        mcx_box     = gr.Textbox(label="MCX Price (Rs./10g)")
        move_1d_box = gr.Textbox(label="1d Expected Move")
        move_7d_box = gr.Textbox(label="7d Expected (%)")
        regime_box  = gr.Textbox(label="Regime")

    with gr.Row():
        var_box       = gr.Textbox(label="VaR Worst Case (Rs.)")
        band_box      = gr.Textbox(label="Confidence Band +/-")
        threshold_box = gr.Textbox(label="Active Threshold")

    reason_box = gr.Textbox(label="Decision Reasoning", lines=2)

    gr.Markdown("### Forecast by Horizon")
    forecast_table = gr.Dataframe(interactive=False)

    gr.Markdown("""
    ---
    *Disclaimer: Decision-support tool only. Combine with inventory levels, cash flow, and market knowledge.*
    """)

    refresh_btn.click(
        fn=get_forecast,
        outputs=[
            signal_box, mcx_box, move_1d_box, move_7d_box,
            var_box, band_box, regime_box, threshold_box,
            reason_box, ts_box, forecast_table,
        ],
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)