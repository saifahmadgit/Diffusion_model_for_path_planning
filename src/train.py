import tensorflow as tf
from tensorflow.keras import optimizers, losses
from config import (
    DATA_DIR,
    IMAGE_SIZE,
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
    dataset = tf.data.Dataset.list_files(DATA_DIR + "/*.png", shuffle=True)
    dataset = dataset.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset


def main():
    dataset = build_dataset()

    ddm = DiffusionModel()
    ddm.normalizer.adapt(dataset)

    ddm.compile(
        optimizer=optimizers.AdamW(
            learning_rate=LEARNING_RATE, weight_decay=WEIGHT_DECAY
        ),
        loss=losses.MeanAbsoluteError(),
    )

    ddm.fit(dataset, epochs=EPOCHS)
    ddm.save_weights("checkpoints/diffusion_model.weights.h5")


if __name__ == "__main__":
    main()
