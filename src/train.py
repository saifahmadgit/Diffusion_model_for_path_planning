import os

import wandb
from wandb.integration.keras import WandbMetricsLogger
import tensorflow as tf

for _gpu in tf.config.list_physical_devices("GPU"):
    tf.config.experimental.set_memory_growth(_gpu, True)

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
    DIFFUSION_STEPS,
    MIN_SIGNAL_RATE,
    MAX_SIGNAL_RATE,
    EMA,
    IMAGE_SIZE,
)
from diffusion_policy import DiffusionModel


def build_dataset():
    def load_image(path):
        raw = tf.io.read_file(path)
        img = tf.image.decode_png(raw, channels=CHANNELS)
        return tf.cast(img, tf.float32) / 255.0

    def load_pair(cond_path, tgt_path):
        return load_image(cond_path), load_image(tgt_path)

    cond_paths = sorted(tf.io.gfile.glob(COND_DIR + "/*.png"))
    tgt_paths  = sorted(tf.io.gfile.glob(DATA_DIR + "/*.png"))
    n = len(cond_paths)

    return (
        tf.data.Dataset.from_tensor_slices((cond_paths, tgt_paths))
        .shuffle(n, seed=42)
        .map(load_pair, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(BATCH_SIZE, drop_remainder=True)
        .prefetch(tf.data.AUTOTUNE)
    ), n


def build_normalizer_dataset():
    """Stream a small subset of targets to adapt the normalizer without OOM."""
    def load_image(path):
        raw = tf.io.read_file(path)
        img = tf.image.decode_png(raw, channels=CHANNELS)
        return tf.cast(img, tf.float32) / 255.0

    tgt_paths = sorted(tf.io.gfile.glob(DATA_DIR + "/*.png"))[:2000]
    return (
        tf.data.Dataset.from_tensor_slices(tgt_paths)
        .map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(BATCH_SIZE)
        .prefetch(tf.data.AUTOTUNE)
    )


def main():
    wandb.init(
        project="diffusion-path-planning",
        config={
            "batch_size": BATCH_SIZE,
            "epochs": EPOCHS,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "image_size": IMAGE_SIZE,
            "channels": CHANNELS,
            "diffusion_steps": DIFFUSION_STEPS,
            "min_signal_rate": MIN_SIGNAL_RATE,
            "max_signal_rate": MAX_SIGNAL_RATE,
            "ema": EMA,
        },
    )

    dataset, n_samples = build_dataset()

    ddm = DiffusionModel()
    # adapt normalizer on a streamed subset of targets — avoids loading all into GPU
    ddm.normalizer.adapt(build_normalizer_dataset())

    ddm.compile(
        optimizer=optimizers.AdamW(
            learning_rate=LEARNING_RATE, weight_decay=WEIGHT_DECAY
        ),
        loss=losses.MeanAbsoluteError(),
    )

    class EpochCheckpoint(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            if (epoch + 1) % 25 == 0:
                self.model.network.save_weights(
                    os.path.join(CHECKPOINT_DIR, f"ckpt_epoch{epoch+1:02d}.weights.h5")
                )
                self.model.ema_network.save_weights(
                    os.path.join(CHECKPOINT_DIR, f"ckpt_ema_epoch{epoch+1:02d}.weights.h5")
                )

    ddm.fit(dataset, epochs=EPOCHS, callbacks=[WandbMetricsLogger(log_freq=10), EpochCheckpoint()])
    ddm.network.save_weights(os.path.join(CHECKPOINT_DIR, "network_final.weights.h5"))
    ddm.ema_network.save_weights(os.path.join(CHECKPOINT_DIR, "ema_network_final.weights.h5"))
    wandb.finish()


if __name__ == "__main__":
    main()
