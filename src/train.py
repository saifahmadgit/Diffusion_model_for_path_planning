import os

import tensorflow as tf
from tensorflow.keras import optimizers, losses
from config import (
    DATA_DIR,
    COND_DIR,
    CHECKPOINT_DIR,
    CHANNELS,
    BATCH_SIZE,
    EPOCHS,
    LEARNING_RATE,
    WEIGHT_DECAY,
)
from diffusion_policy import DiffusionModel


def load_image(file_path):
    raw = tf.io.read_file(file_path)
    image = tf.image.decode_png(raw, channels=CHANNELS)
    image = tf.cast(image, tf.float32)
    image = image / 255.0
    return image


def build_dataset():
    cond_files = tf.data.Dataset.list_files(COND_DIR + "/*.png", shuffle=False)
    target_files = tf.data.Dataset.list_files(DATA_DIR + "/*.png", shuffle=False)

    dataset = tf.data.Dataset.zip((cond_files, target_files))
    dataset = dataset.shuffle(buffer_size=10000, seed=42)
    dataset = dataset.map(
        lambda c, t: (load_image(c), load_image(t)),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset


def main():
    dataset = build_dataset()

    ddm = DiffusionModel()
    # adapt normalizer on target images only (condition images are not denoised)
    ddm.normalizer.adapt(dataset.map(lambda cond, tgt: tgt))

    ddm.compile(
        optimizer=optimizers.AdamW(
            learning_rate=LEARNING_RATE, weight_decay=WEIGHT_DECAY
        ),
        loss=losses.MeanAbsoluteError(),
    )

    ddm.fit(dataset, epochs=EPOCHS)
    ddm.save_weights(os.path.join(CHECKPOINT_DIR, "diffusion_model.weights.h5"))


if __name__ == "__main__":
    main()
