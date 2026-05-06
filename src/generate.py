import argparse
import os
from datetime import datetime

import numpy as np
import tensorflow as tf

for _gpu in tf.config.list_physical_devices("GPU"):
    tf.config.experimental.set_memory_growth(_gpu, True)

from PIL import Image

from config import (
    CHANNELS,
    CHECKPOINT_DIR,
    COND_DIR,
    DIFFUSION_STEPS,
    SAMPLES_DIR,
)
from diffusion_policy import DiffusionModel


def list_runs(checkpoint_dir):
    runs = sorted(
        d for d in os.listdir(checkpoint_dir)
        if os.path.isdir(os.path.join(checkpoint_dir, d))
    )
    return runs


def resolve_checkpoint(run_dir):
    final = os.path.join(run_dir, "ema_network_final.weights.h5")
    if os.path.isfile(final):
        return final
    periodic = sorted(
        f for f in os.listdir(run_dir) if f.startswith("ckpt_ema_epoch")
    )
    if periodic:
        return os.path.join(run_dir, periodic[-1])
    raise FileNotFoundError(f"No EMA checkpoint found in {run_dir}")


def parse_args():
    parser = argparse.ArgumentParser(description="Generate paths from a trained checkpoint.")
    parser.add_argument(
        "--checkpoint", "-c",
        help=(
            "Run folder name (e.g. 20260505_143021) inside CHECKPOINT_DIR, "
            "or an absolute path to a .h5 weights file. "
            "Omit to list available runs."
        ),
        default=None,
    )
    parser.add_argument(
        "--num-images", "-n",
        type=int,
        default=10,
        help="Number of images to generate (default: 10).",
    )
    return parser.parse_args()


def load_images_random(directory, num_images):
    all_files = sorted(
        f for f in os.listdir(directory) if f.lower().endswith(".png")
    )
    chosen = np.random.choice(all_files, size=num_images, replace=False)
    images, paths = [], []
    for fname in chosen:
        path = os.path.join(directory, fname)
        raw = tf.io.read_file(path)
        img = tf.image.decode_png(raw, channels=CHANNELS)
        img = tf.cast(img, tf.float32) / 255.0
        images.append(img)
        paths.append(path)
    return tf.stack(images), paths


def save_images(images, folder, prefix="generated"):
    os.makedirs(folder, exist_ok=True)
    for i, image in enumerate(images):
        image = (image.numpy() * 255).astype(np.uint8)
        Image.fromarray(image).save(os.path.join(folder, f"{prefix}_{i}.png"))


def main():
    args = parse_args()

    if args.checkpoint is None:
        runs = list_runs(CHECKPOINT_DIR)
        if runs:
            print("Available checkpoint runs:")
            for r in runs:
                print(f"  {r}")
        else:
            print(f"No checkpoint runs found in {CHECKPOINT_DIR}")
        return

    if os.path.isfile(args.checkpoint):
        checkpoint_path = args.checkpoint
    else:
        run_dir = (
            args.checkpoint
            if os.path.isabs(args.checkpoint)
            else os.path.join(CHECKPOINT_DIR, args.checkpoint)
        )
        checkpoint_path = resolve_checkpoint(run_dir)

    print(f"Loading checkpoint: {checkpoint_path}")

    out_dir = os.path.join(SAMPLES_DIR, datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(out_dir, exist_ok=True)

    condition_images, cond_paths = load_images_random(COND_DIR, args.num_images)

    cond_out_dir = os.path.join(out_dir, "conditions")
    os.makedirs(cond_out_dir, exist_ok=True)
    for i, src in enumerate(cond_paths):
        img = Image.open(src)
        img.save(os.path.join(cond_out_dir, f"condition_{i}.png"))
    print(f"Saved condition images to {cond_out_dir}")

    # normalize condition to [-1, 1] — same as what train_step does before feeding to UNet
    condition_images = condition_images * 2.0 - 1.0

    ddm = DiffusionModel()
    ddm.ema_network.load_weights(checkpoint_path)

    generated, snapshots = ddm.generate(
        condition_images=condition_images,
        diffusion_steps=DIFFUSION_STEPS,
    )

    save_images(generated, out_dir)
    print(f"Saved {len(generated)} images to {out_dir}")

    strips_dir = os.path.join(out_dir, "strips")
    os.makedirs(strips_dir, exist_ok=True)
    for i in range(args.num_images):
        frames = [snapshots[s][i].numpy() for s in range(len(snapshots))]
        strip = np.concatenate(frames, axis=1)
        strip = (strip * 255).astype(np.uint8)
        Image.fromarray(strip).save(os.path.join(strips_dir, f"strip_{i}.png"))
    print(f"Saved denoising strips to {strips_dir}")


if __name__ == "__main__":
    main()
