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
    """Map diffusion time in [0, 1] to noise and signal rates via a cosine schedule."""
    min_signal_rate = MIN_SIGNAL_RATE
    max_signal_rate = MAX_SIGNAL_RATE
    start_angle = tf.acos(max_signal_rate)
    end_angle = tf.acos(min_signal_rate)

    diffusion_angles = start_angle + diffusion_times * (end_angle - start_angle)

    signal_rates = tf.cos(diffusion_angles)
    noise_rates = tf.sin(diffusion_angles)

    return noise_rates, signal_rates


def sinusoidal_embedding(x):
    """Encode a scalar noise variance into a sinusoidal frequency embedding."""
    frequencies = tf.exp(
        tf.linspace(tf.math.log(1.0), tf.math.log(1000.0), NOISE_EMBEDDING_SIZE // 2)
    )
    angular_speeds = 2.0 * math.pi * frequencies
    embeddings = tf.concat(
        [tf.sin(angular_speeds * x), tf.cos(angular_speeds * x)], axis=3
    )
    return embeddings


def ResidualBlock(width, noise_emb):
    """Conv residual block that injects the noise embedding into every feature map."""
    def apply(x):
        input_width = x.shape[3]
        if input_width == width:
            residual = x
        else:
            residual = layers.Conv2D(width, kernel_size=1)(x)
        x = layers.BatchNormalization(center=False, scale=False)(x)
        x = layers.Conv2D(width, kernel_size=3, padding="same", activation=activations.swish)(x)

        # dense layer is to mathc the dimesion so that it can be added
        noise_proj = layers.Dense(width)(noise_emb)
        x = layers.Add()([x, noise_proj])

        x = layers.Conv2D(width, kernel_size=3, padding="same")(x)
        x = layers.Add()([x, residual])
        return x

    return apply


def AttentionBlock(num_heads=4):
    """Self-attention block that captures long-range spatial dependencies."""
    def apply(x):
        _, H, W, C = x.shape
        h = layers.LayerNormalization()(x)
        h = layers.Reshape((H * W, C))(h)
        h = layers.MultiHeadAttention(num_heads=num_heads, key_dim=C // num_heads)(h, h)
        h = layers.Reshape((H, W, C))(h)
        return layers.Add()([x, h])

    return apply


def DownBlock(width, block_depth, noise_emb, use_attention=False, num_heads=4):
    """Encoder block: applies residual (and optionally attention) layers then downsamples."""
    def apply(x):
        x, skips = x
        for _ in range(block_depth):
            x = ResidualBlock(width, noise_emb)(x)
            if use_attention:
                x = AttentionBlock(num_heads)(x)
            skips.append(x)
        x = layers.AveragePooling2D(pool_size=2)(x)
        return x

    return apply


def UpBlock(width, block_depth, noise_emb, use_attention=False, num_heads=4):
    """Decoder block: upsamples then applies residual (and optionally attention) layers with skip connections."""
    def apply(x):
        x, skips = x
        x = layers.UpSampling2D(size=2, interpolation="bilinear")(x)
        for _ in range(block_depth):
            x = layers.Concatenate()([x, skips.pop()])
            x = ResidualBlock(width, noise_emb)(x)
            if use_attention:
                x = AttentionBlock(num_heads)(x)
        return x

    return apply


def build_unet():
    """Build the noise-conditioned U-Net that predicts noise from a noisy image and condition map."""
    # Input 1: noisy target (3ch) concatenated with condition image (3ch) = 6ch
    noisy_images = layers.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, CHANNELS * 2))
    x = layers.Conv2D(32, kernel_size=1)(noisy_images)

    # Input 2: noise level as a scalar, shape (batch, 1, 1, 1)
    # convert it to a 32-dim embedding, shape stays (batch, 1, 1, 32)
    # this gets passed into every ResidualBlock so all layers know the noise level
    noise_variances = layers.Input(shape=(1, 1, 1))
    noise_emb = layers.Lambda(sinusoidal_embedding)(noise_variances)

    # Encoder
    skips = []
    x = DownBlock(32, block_depth=2, noise_emb=noise_emb)([x, skips])
    x = DownBlock(64, block_depth=2, noise_emb=noise_emb)([x, skips])
    x = DownBlock(96, block_depth=2, noise_emb=noise_emb, use_attention=True, num_heads=4)([x, skips])  # 32×32

    # Bottleneck (16×16): ResBlock → Attention → ResBlock, following DDPM/ADM
    x = ResidualBlock(128, noise_emb)(x)
    x = AttentionBlock(num_heads=4)(x)
    x = ResidualBlock(128, noise_emb)(x)

    # Decoder
    x = UpBlock(96, block_depth=2, noise_emb=noise_emb, use_attention=True, num_heads=4)([x, skips])  # 32×32
    x = UpBlock(64, block_depth=2, noise_emb=noise_emb)([x, skips])
    x = UpBlock(32, block_depth=2, noise_emb=noise_emb)([x, skips])

    # Output
    x = layers.Conv2D(CHANNELS, kernel_size=1, kernel_initializer="zeros")(x)

    return models.Model([noisy_images, noise_variances], x, name="unet")


class DiffusionModel(models.Model):
    """Denoising diffusion model with an EMA copy of the network for stable inference."""

    def __init__(self):
        super().__init__()

        self.network = build_unet()
        self.ema_network = build_unet()
        self.ema_network.set_weights(self.network.get_weights())
        self.diffusion_schedule = offset_cosine_diffusion_schedule

    def compile(self, **kwargs):
        """Set up optimizer, loss, and metrics."""
        super().compile(**kwargs)
        self.noise_loss_tracker = metrics.Mean(name="n_loss")

    @property
    def metrics(self):
        return [self.noise_loss_tracker]

    def denormalize(self, images):
        # reverse the x * 2 - 1 normalization: [-1, 1] → [0, 1]
        images = (images + 1.0) / 2.0
        return tf.clip_by_value(images, 0.0, 1.0)

    def denoise(self, noisy_images, noise_rates, signal_rates, training, condition):
        """Predict noise and clean image from a noisy image at a given noise level."""
        if training:
            network = self.network
        else:
            network = self.ema_network
        unet_input = tf.concat([noisy_images, condition], axis=-1)
        pred_noises = network([unet_input, noise_rates**2], training=training)
        pred_images = (noisy_images - noise_rates * pred_noises) / signal_rates
        return pred_noises, pred_images

    def train_step(self, data):
        """Single training step: corrupt target, predict noise, update weights and EMA."""
        condition, target = data
        # normalize both to [-1, 1] so they are on the same scale when concatenated
        condition = condition * 2.0 - 1.0
        target = target * 2.0 - 1.0
        batch_size = tf.shape(target)[0]
        # noise generation, at other end of original image, complete noise, it is also the output ground truth which network learns to predict
        noises = tf.random.normal(shape=(batch_size, IMAGE_SIZE, IMAGE_SIZE, CHANNELS))
        # random number between 0 to 1 to decide how much noise to add, this goes to scheduler
        diffusion_times = tf.random.uniform(
            shape=(batch_size, 1, 1, 1), minval=0.0, maxval=1.0
        )
        # getting noise rates and signal rates by inputting random time (0,1)
        noise_rates, signal_rates = self.diffusion_schedule(diffusion_times)
        # calculate noisy image
        noisy_images = signal_rates * target + noise_rates * noises

        with tf.GradientTape() as tape:
            pred_noises, pred_images = self.denoise(
                noisy_images, noise_rates, signal_rates, training=True, condition=condition
            )
            noise_loss = self.loss(noises, pred_noises)

        gradients = tape.gradient(noise_loss, self.network.trainable_weights)
        self.optimizer.apply_gradients(zip(gradients, self.network.trainable_weights))
        self.noise_loss_tracker.update_state(noise_loss)

        for weight, ema_weight in zip(self.network.weights, self.ema_network.weights):
            ema_weight.assign(EMA * ema_weight + (1 - EMA) * weight)

        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        """Validation step: compute noise prediction loss without updating weights."""
        condition, target = data
        condition = condition * 2.0 - 1.0
        target = target * 2.0 - 1.0
        batch_size = tf.shape(target)[0]
        noises = tf.random.normal(shape=(batch_size, IMAGE_SIZE, IMAGE_SIZE, CHANNELS))
        diffusion_times = tf.random.uniform(
            shape=(batch_size, 1, 1, 1), minval=0.0, maxval=1.0
        )
        noise_rates, signal_rates = self.diffusion_schedule(diffusion_times)
        noisy_images = signal_rates * target + noise_rates * noises
        pred_noises, pred_images = self.denoise(
            noisy_images, noise_rates, signal_rates, training=False, condition=condition
        )
        noise_loss = self.loss(noises, pred_noises)
        self.noise_loss_tracker.update_state(noise_loss)
        return {m.name: m.result() for m in self.metrics}

    def generate(self, condition_images, diffusion_steps, initial_noise=None):
        """Run reverse diffusion from pure noise to a generated path image."""
        num_images = condition_images.shape[0]
        if initial_noise is None:
            initial_noise = tf.random.normal(
                shape=(num_images, IMAGE_SIZE, IMAGE_SIZE, CHANNELS)
            )
        generated_images, snapshots = self.reverse_diffusion(initial_noise, condition_images, diffusion_steps)
        generated_images = self.denormalize(generated_images)
        return generated_images, snapshots

    def reverse_diffusion(self, initial_noise, condition_images, diffusion_steps):
        """Iteratively denoise from t=1 to t=0 using DDPM sampling."""
        num_images = initial_noise.shape[0]
        step_size = 1.0 / diffusion_steps
        current_images = initial_noise
        save_every = diffusion_steps // 10
        snapshots = []
        for step in range(diffusion_steps):
            diffusion_times = tf.ones((num_images, 1, 1, 1)) - step * step_size
            noise_rates, signal_rates = self.diffusion_schedule(diffusion_times)
            pred_noises, pred_images = self.denoise(
                current_images, noise_rates, signal_rates, training=False, condition=condition_images
            )
            next_diffusion_times = diffusion_times - step_size
            next_noise_rates, next_signal_rates = self.diffusion_schedule(
                next_diffusion_times
            )
            current_images = (
                next_signal_rates * pred_images + next_noise_rates * pred_noises
            )
            if (step + 1) % save_every == 0:
                snapshots.append(self.denormalize(pred_images))
        return pred_images, snapshots
