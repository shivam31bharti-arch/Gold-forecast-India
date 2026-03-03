import gradio as gr
import json
from inference.predictor import predict_all_horizons
from inference.decision_engine import generate_signal
from utils.config_loader import load_config
import pandas as pd

# Load config
cfg = load_config("config.yaml")

def run_prediction():
    # Load latest processed features
    df = pd.read_parquet("data/processed/features_daily.parquet")

    preds = predict_all_horizons(df, cfg)
    signal = generate_signal(preds, cfg)

    return json.dumps(signal, indent=2)

with gr.Blocks() as demo:
    gr.Markdown("# 🇮🇳 Gold Forecast India")
    gr.Markdown("AI-based MCX Gold Forecast for Jewellery Sellers")

    output = gr.Textbox(label="Prediction Output", lines=20)
    btn = gr.Button("Run Forecast")

    btn.click(fn=run_prediction, outputs=output)

demo.launch()