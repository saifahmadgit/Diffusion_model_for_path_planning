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
            self.model.network.save_weights(
                os.path.join(CHECKPOINT_DIR, f"ckpt_epoch{epoch+1:02d}.weights.h5")
            )
            self.model.ema_network.save_weights(
                os.path.join(CHECKPOINT_DIR, f"ckpt_ema_epoch{epoch+1:02d}.weights.h5")
            )

    ddm.fit(dataset, epochs=EPOCHS, callbacks=[WandbMetricsLogger(), EpochCheckpoint()])
    ddm.network.save_weights(os.path.join(CHECKPOINT_DIR, "network_final.weights.h5"))
    ddm.ema_network.save_weights(os.path.join(CHECKPOINT_DIR, "ema_network_final.weights.h5"))
    wandb.finish()


if __name__ == "__main__":
    main()
