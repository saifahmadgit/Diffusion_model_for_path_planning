import os

import numpy as np
import tensorflow as tf
from PIL import Image

from config import (
    CHANNELS,
    CHECKPOINT_DIR,
    COND_DIR,
    DIFFUSION_STEPS,
    SAMPLES_DIR,
)
from diffusion_policy import DiffusionModel


def load_condition_images(cond_dir, num_images):
    files = sorted(os.listdir(cond_dir))[:num_images]
    images = []
    for fname in files:
        raw = tf.io.read_file(os.path.join(cond_dir, fname))
        img = tf.image.decode_png(raw, channels=CHANNELS)
        img = tf.cast(img, tf.float32) / 255.0
        images.append(img)
    return tf.stack(images)


def save_images(images, folder):
    os.makedirs(folder, exist_ok=True)
    for i, image in enumerate(images):
        image = (image.numpy() * 255).astype(np.uint8)
        img = Image.fromarray(image)
        img.save(os.path.join(folder, f"generated_{i}.png"))


def main():
    num_images = 4
    condition_images = load_condition_images(COND_DIR, num_images)

    ddm = DiffusionModel()
    ddm.load_weights(os.path.join(CHECKPOINT_DIR, "diffusion_model.weights.h5"))

    generated = ddm.generate(
        condition_images=condition_images,
        diffusion_steps=DIFFUSION_STEPS,
    )

    save_images(generated, SAMPLES_DIR)
    print(f"Saved {len(generated)} images to {SAMPLES_DIR}")


if __name__ == "__main__":
    main()
