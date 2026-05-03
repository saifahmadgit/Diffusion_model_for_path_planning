import yaml


def load_config(path="config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)


cfg = load_config()

DATA_DIR = cfg["paths"]["data_dir"]
IMAGE_SIZE = cfg["image"]["size"]
CHANNELS = cfg["image"]["channels"]
