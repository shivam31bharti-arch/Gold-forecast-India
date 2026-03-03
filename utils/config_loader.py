"""
utils/config_loader.py — Load config.yaml into a dict.
"""
import yaml
import os


def load_config(path: str = None) -> dict:
    if path is None:
        # Walk up from this file's location to find config.yaml
        base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        path = os.path.join(base, "config.yaml")
    with open(path, "r") as f:
        return yaml.safe_load(f)
