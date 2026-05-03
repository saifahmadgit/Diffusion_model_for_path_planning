# config.py

import os

import yaml


def load_config(path="config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)


cfg = load_config()

DATA_DIR = cfg["paths"]["data_dir"]
CHECKPOINT_DIR = cfg["paths"]["checkpoint_dir"]
SAMPLES_DIR = cfg["paths"]["samples_dir"]

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(SAMPLES_DIR, exist_ok=True)

IMAGE_SIZE = cfg["image"]["size"]
CHANNELS = cfg["image"]["channels"]

BATCH_SIZE = cfg["training"]["batch_size"]
EPOCHS = cfg["training"]["epochs"]

MIN_SIGNAL_RATE = cfg["diffusion"]["min_signal_rate"]
MAX_SIGNAL_RATE = cfg["diffusion"]["max_signal_rate"]

NOISE_EMBEDDING_SIZE = cfg["unet"]["noise_embedding_size"]
