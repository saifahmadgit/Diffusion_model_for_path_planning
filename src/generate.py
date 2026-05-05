import os

import numpy as np
import tensorflow as tf

for _gpu in tf.config.list_physical_devices("GPU"):
    tf.config.experimental.set_memory_growth(_gpu, True)

from PIL import Image

from config import (
    CHANNELS,
    CHECKPOINT_DIR,
    COND_DIR,
    DATA_DIR,
    DIFFUSION_STEPS,
    SAMPLES_DIR,
    BATCH_SIZE,
)
from diffusion_policy import DiffusionModel


def latest_ema_checkpoint(checkpoint_dir):
    files = [f for f in os.listdir(checkpoint_dir) if f.startswith("ckpt_ema_epoch")]
    if not files:
        return os.path.join(checkpoint_dir, "ema_network_final.weights.h5")
    files.sort()
    return os.path.join(checkpoint_dir, files[-1])


def load_images(directory, num_images):
    files = sorted(os.listdir(directory))[:num_images]
    images = []
    for fname in files:
        raw = tf.io.read_file(os.path.join(directory, fname))
        img = tf.image.decode_png(raw, channels=CHANNELS)
        img = tf.cast(img, tf.float32) / 255.0
        images.append(img)
    return tf.stack(images)


def save_images(images, folder):
    os.makedirs(folder, exist_ok=True)
    for i, image in enumerate(images):
        image = (image.numpy() * 255).astype(np.uint8)
        Image.fromarray(image).save(os.path.join(folder, f"generated_{i}.png"))


def main():
    num_images = 10

    checkpoint_path = latest_ema_checkpoint(CHECKPOINT_DIR)
    print(f"Loading checkpoint: {checkpoint_path}")

    condition_images = load_images(COND_DIR, num_images)

    # Re-adapt normalizer on a small subset of training targets so
    # denormalize() has the correct mean and variance.
    target_subset = load_images(DATA_DIR, BATCH_SIZE)
    ddm = DiffusionModel()
    ddm.normalizer.adapt(target_subset)

    ddm.ema_network.load_weights(checkpoint_path)

    generated, snapshots = ddm.generate(
        condition_images=condition_images,
        diffusion_steps=DIFFUSION_STEPS,
    )

    save_images(generated, SAMPLES_DIR)
    print(f"Saved {len(generated)} images to {SAMPLES_DIR}")

    # save one horizontal strip per image showing 10 denoising steps
    strips_dir = os.path.join(SAMPLES_DIR, "strips")
    os.makedirs(strips_dir, exist_ok=True)
    for i in range(num_images):
        frames = [snapshots[s][i].numpy() for s in range(len(snapshots))]
        strip = np.concatenate(frames, axis=1)          # (128, 128*10, 3)
        strip = (strip * 255).astype(np.uint8)
        Image.fromarray(strip).save(os.path.join(strips_dir, f"strip_{i}.png"))
    print(f"Saved denoising strips to {strips_dir}")


if __name__ == "__main__":
    main()
