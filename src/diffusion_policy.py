import tensorflow as tf
from tensorflow.keras import layers, metrics, models
from config import (
    IMAGE_SIZE, CHANNELS,
    MIN_SIGNAL_RATE, MAX_SIGNAL_RATE,
    EMA
)



def offset_cosine_diffusion_schedule(diffusion_times):
    min_signal_rate = MIN_SIGNAL_RATE    
    max_signal_rate = MAX_SIGNAL_RATE    
    start_angle = tf.acos(max_signal_rate)
    end_angle = tf.acos(min_signal_rate)

    diffusion_angles = start_angle + diffusion_times * (end_angle - start_angle)

    signal_rates = tf.cos(diffusion_angles)
    noise_rates  = tf.sin(diffusion_angles)

    return noise_rates, signal_rates

def sinusoidal_embedding(x):
    pass


def ResidualBlock(width):
    pass


def DownBlock(width, block_depth):
    pass


def UpBlock(width, block_depth):
    pass


def build_unet():
    pass


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
