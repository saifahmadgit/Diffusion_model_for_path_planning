import os
import yaml


def load_config():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(base_dir, "config.yaml")
    with open(path, "r") as f:
        return yaml.safe_load(f)


cfg = load_config()

_base = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.normpath(os.path.join(_base, cfg["paths"]["data_dir"]))
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
EMA = cfg["diffusion"]["ema"]

NOISE_EMBEDDING_SIZE = cfg["unet"]["noise_embedding_size"]

LEARNING_RATE = cfg["training"]["learning_rate"]
WEIGHT_DECAY = cfg["training"]["weight_decay"]
