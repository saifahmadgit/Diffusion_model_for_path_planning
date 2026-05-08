import argparse
import os
from datetime import datetime

import wandb
from wandb.integration.keras import WandbMetricsLogger
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
    DIFFUSION_STEPS,
    MIN_SIGNAL_RATE,
    MAX_SIGNAL_RATE,
    EMA,
    IMAGE_SIZE,
    USE_WANDB,
)
from diffusion_policy import DiffusionModel


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help=f"Batch size (default: {BATCH_SIZE} from config)")
    return parser.parse_args()


def configure_gpu():
    """Enable memory growth on all GPUs to avoid pre-allocating the full VRAM."""
    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)


def load_image(path):
    """Load a PNG from disk and normalize pixel values to [0, 1]."""
    raw = tf.io.read_file(path)
    img = tf.image.decode_png(raw, channels=CHANNELS)
    img = tf.cast(img, tf.float32) / 255.0
    return img


def load_pair(cond_path, tgt_path):
    """Load a matched condition and target image pair."""
    condition = load_image(cond_path)
    target = load_image(tgt_path)
    return condition, target


def build_dataset(batch_size):
    """Build a shuffled, batched, prefetched tf.data pipeline from the condition/target image dirs."""
    # returns tensorflow pipeline object which has information of how to parse through data in batches when iterated over
    cond_paths = sorted(tf.io.gfile.glob(COND_DIR + "/*.png"))
    tgt_paths = sorted(tf.io.gfile.glob(DATA_DIR + "/*.png"))
    n = len(cond_paths)

    dataset = tf.data.Dataset.from_tensor_slices((cond_paths, tgt_paths))
    dataset = dataset.shuffle(n, seed=42)
    dataset = dataset.map(load_pair, num_parallel_calls=4)
    dataset = dataset.batch(batch_size, drop_remainder=True)
    dataset = dataset.prefetch(4)
    return dataset, n



class EpochCheckpoint(tf.keras.callbacks.Callback):
    def __init__(self, run_dir):
        super().__init__()
        self.run_dir = run_dir

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % 25 == 0:
            network_path = os.path.join(self.run_dir, f"ckpt_epoch{epoch+1:02d}.weights.h5")
            ema_path = os.path.join(self.run_dir, f"ckpt_ema_epoch{epoch+1:02d}.weights.h5")
            self.model.network.save_weights(network_path)
            self.model.ema_network.save_weights(ema_path)


def main():
    args = parse_args()
    batch_size = args.batch_size

    configure_gpu()

    if USE_WANDB:
        wandb.init(
            project="diffusion-path-planning",
            config={
                "batch_size": batch_size,
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

    run_dir = os.path.join(CHECKPOINT_DIR, datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(run_dir, exist_ok=True)

    dataset, n_samples = build_dataset(batch_size)

    ddm = DiffusionModel()

    optimizer = optimizers.AdamW(learning_rate=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    ddm.compile(optimizer=optimizer, loss=losses.MeanAbsoluteError())

    callbacks = [EpochCheckpoint(run_dir)]
    if USE_WANDB:
        callbacks.insert(0, WandbMetricsLogger(log_freq=10))

    ddm.fit(dataset, epochs=EPOCHS, callbacks=callbacks, verbose=1)

    ddm.network.save_weights(os.path.join(run_dir, "network_final.weights.h5"))
    ddm.ema_network.save_weights(os.path.join(run_dir, "ema_network_final.weights.h5"))

    if USE_WANDB:
        wandb.finish()


if __name__ == "__main__":
    main()
