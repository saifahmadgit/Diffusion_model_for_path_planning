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

    cond_paths = sorted(tf.io.gfile.glob(COND_DIR + "/*.png"))
    tgt_paths  = sorted(tf.io.gfile.glob(DATA_DIR + "/*.png"))

    # load every image into memory upfront as (N, 128, 128, 3) tensors
    cond_images = tf.stack([load_image(p) for p in cond_paths])
    tgt_images  = tf.stack([load_image(p) for p in tgt_paths])

    return (
        tf.data.Dataset.from_tensor_slices((cond_images, tgt_images))
        .shuffle(len(cond_paths), seed=42)
        .batch(BATCH_SIZE, drop_remainder=True)
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
