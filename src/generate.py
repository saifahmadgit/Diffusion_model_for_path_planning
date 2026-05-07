import argparse
import os
from datetime import datetime

import numpy as np
import tensorflow as tf

for _gpu in tf.config.list_physical_devices("GPU"):
    tf.config.experimental.set_memory_growth(_gpu, True)

from PIL import Image, ImageDraw, ImageFont

from config import (
    CHANNELS,
    CHECKPOINT_DIR,
    COND_DIR,
    DATA_DIR,
    DIFFUSION_STEPS,
    SAMPLES_DIR,
)
from diffusion_policy import DiffusionModel

LABEL_HEIGHT = 24
LABEL_FONT_SIZE = 14


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
        (f for f in os.listdir(run_dir) if f.startswith("ckpt_ema_epoch")),
        key=lambda f: int(f.replace("ckpt_ema_epoch", "").replace(".weights.h5", ""))
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


def load_image_tf(path):
    raw = tf.io.read_file(path)
    img = tf.image.decode_png(raw, channels=CHANNELS)
    return tf.cast(img, tf.float32) / 127.5 - 1.0


def load_images_random(cond_dir, target_dir, num_images):
    all_files = sorted(
        f for f in os.listdir(cond_dir) if f.lower().endswith(".png")
    )
    chosen = np.random.choice(all_files, size=num_images, replace=False)
    cond_images, gt_images, fnames = [], [], []
    for fname in chosen:
        cond_images.append(load_image_tf(os.path.join(cond_dir, fname)))
        gt_images.append(load_image_tf(os.path.join(target_dir, fname)))
        fnames.append(fname)
    return tf.stack(cond_images), tf.stack(gt_images), fnames


def _make_label_bar(width, label, bg=(240, 240, 240), fg=(30, 30, 30)):
    bar = Image.new("RGB", (width, LABEL_HEIGHT), bg)
    draw = ImageDraw.Draw(bar)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", LABEL_FONT_SIZE)
    except Exception:
        font = ImageFont.load_default()
    bbox = draw.textbbox((0, 0), label, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]
    draw.text(((width - text_w) // 2, (LABEL_HEIGHT - text_h) // 2), label, fill=fg, font=font)
    return bar


def save_comparison(cond_img, gen_img, gt_img, path):
    """Save a horizontal [Condition | Generated | Ground Truth] composite with labels."""
    panels = [
        ("Condition", cond_img),
        ("Generated", gen_img),
        ("Ground Truth", gt_img),
    ]
    imgs = []
    for label, arr in panels:
        arr_uint8 = (np.clip(arr, 0.0, 1.0) * 255).astype(np.uint8)
        content = Image.fromarray(arr_uint8)
        w, h = content.size
        bar = _make_label_bar(w, label)
        panel = Image.new("RGB", (w, LABEL_HEIGHT + h))
        panel.paste(bar, (0, 0))
        panel.paste(content, (0, LABEL_HEIGHT))
        imgs.append(panel)

    gap = 8
    total_w = sum(img.width for img in imgs) + gap * (len(imgs) - 1)
    total_h = imgs[0].height
    composite = Image.new("RGB", (total_w, total_h), (200, 200, 200))
    x = 0
    for img in imgs:
        composite.paste(img, (x, 0))
        x += img.width + gap
    composite.save(path)


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

    condition_images, gt_images, fnames = load_images_random(COND_DIR, DATA_DIR, args.num_images)

    ddm = DiffusionModel()
    ddm.ema_network.load_weights(checkpoint_path)

    # condition_images is already in [-1, 1]; generate() returns [0, 1] via denormalize()
    generated, _ = ddm.generate(
        condition_images=condition_images,
        diffusion_steps=DIFFUSION_STEPS,
    )

    # convert cond and gt from [-1, 1] → [0, 1] for display
    cond_display = (condition_images + 1.0) / 2.0
    gt_display = (gt_images + 1.0) / 2.0

    for i in range(args.num_images):
        save_comparison(
            cond_img=cond_display[i].numpy(),
            gen_img=generated[i].numpy(),
            gt_img=gt_display[i].numpy(),
            path=os.path.join(out_dir, f"comparison_{i:03d}.png"),
        )

    print(f"Saved {args.num_images} comparison images to {out_dir}")


if __name__ == "__main__":
    main()
