import math
import tensorflow as tf
from tensorflow.keras import layers, metrics, models, activations
from config import (
    MIN_SIGNAL_RATE,
    MAX_SIGNAL_RATE,
    NOISE_EMBEDDING_SIZE,
    IMAGE_SIZE,
    CHANNELS,
    EMA,
    BATCH_SIZE,
)


def offset_cosine_diffusion_schedule(diffusion_times):
    min_signal_rate = MIN_SIGNAL_RATE
    max_signal_rate = MAX_SIGNAL_RATE
    start_angle = tf.acos(max_signal_rate)
    end_angle = tf.acos(min_signal_rate)

    diffusion_angles = start_angle + diffusion_times * (end_angle - start_angle)

    signal_rates = tf.cos(diffusion_angles)
    noise_rates = tf.sin(diffusion_angles)

    return noise_rates, signal_rates


def sinusoidal_embedding(x):
    frequencies = tf.exp(
        tf.linspace(tf.math.log(1.0), tf.math.log(1000.0), NOISE_EMBEDDING_SIZE // 2)
    )
    angular_speeds = 2.0 * math.pi * frequencies
    embeddings = tf.concat(
        [tf.sin(angular_speeds * x), tf.cos(angular_speeds * x)], axis=3
    )
    return embeddings


def ResidualBlock(width):
    def apply(x):
        input_width = x.shape[3]
        if input_width == width:
            residual = x
        else:
            residual = layers.Conv2D(width, kernel_size=1)(x)
        x = layers.BatchNormalization(center=False, scale=False)(x)
        x = layers.Conv2D(
            width, kernel_size=3, padding="same", activation=activations.swish
        )(x)
        x = layers.Conv2D(width, kernel_size=3, padding="same")(x)
        x = layers.Add()([x, residual])
        return x

    return apply


def DownBlock(width, block_depth):
    def apply(x):
        x, skips = x
        for _ in range(block_depth):
            x = ResidualBlock(width)(x)
            skips.append(x)
        x = layers.AveragePooling2D(pool_size=2)(x)
        return x

    return apply


def UpBlock(width, block_depth):
    def apply(x):
        x, skips = x
        x = layers.UpSampling2D(size=2, interpolation="bilinear")(x)
        for _ in range(block_depth):
            x = layers.Concatenate()([x, skips.pop()])
            x = ResidualBlock(width)(x)
        return x

    return apply


def build_unet():
    #Input 1
    noisy_images = layers.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, CHANNELS))
    x = layers.Conv2D(32, kernel_size=1)(noisy_images)

    #Input 2
    noise_variances = layers.Input(shape=(1, 1, 1))
    noise_embedding = layers.Lambda(sinusoidal_embedding)(noise_variances)
    noise_embedding = layers.UpSampling2D(size=IMAGE_SIZE, interpolation="nearest")(noise_embedding)

    # Merge both inputs
    x = layers.Concatenate()([x, noise_embedding])

    # Encoder
    skips = []
    x = DownBlock(32, block_depth=2)([x, skips])
    x = DownBlock(64, block_depth=2)([x, skips])
    x = DownBlock(96, block_depth=2)([x, skips])

    #Bottleneck 
    x = ResidualBlock(128)(x)
    x = ResidualBlock(128)(x)

    #Decoder
    x = UpBlock(96, block_depth=2)([x, skips])
    x = UpBlock(64, block_depth=2)([x, skips])
    x = UpBlock(32, block_depth=2)([x, skips])

    # Output
    x = layers.Conv2D(CHANNELS, kernel_size=1, kernel_initializer="zeros")(x)

    return models.Model([noisy_images, noise_variances], x, name="unet")


class DiffusionModel(models.Model):
    def __init__(self):
        pass

    def compile(self, **kwargs):
        pass

    @property
    def metrics(self):
        pass

    def denormalize(self, images):
        pass

    def denoise(self, noisy_images, noise_rates, signal_rates, training):
        pass

    def train_step(self, images):
        pass

    def test_step(self, images):
        pass

    def reverse_diffusion(self, initial_noise, diffusion_steps):
        pass

    def generate(self, num_images, diffusion_steps, initial_noise=None):
        pass
