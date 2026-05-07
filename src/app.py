import os
import sys

import numpy as np
import tensorflow as tf

for _gpu in tf.config.list_physical_devices("GPU"):
    tf.config.experimental.set_memory_growth(_gpu, True)

from PIL import Image
import gradio as gr

_SRC = os.path.dirname(os.path.abspath(__file__))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

os.environ.setdefault("GRADIO_TEMP_DIR", os.path.join(os.path.expanduser("~"), ".gradio_tmp"))

from config import CHECKPOINT_DIR, CHANNELS, IMAGE_SIZE
from diffusion_policy import DiffusionModel

_model = None


def _find_checkpoint():
    ckpt = os.path.join(CHECKPOINT_DIR, "20260507_021059", "ckpt_ema_epoch75.weights.h5")
    if not os.path.isfile(ckpt):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")
    return ckpt


def _get_model():
    global _model
    if _model is None:
        ckpt = _find_checkpoint()
        print(f"[PathForge] Loading checkpoint: {ckpt}")
        _model = DiffusionModel()
        _model.ema_network.load_weights(ckpt)
        print("[PathForge] Model ready.")
    return _model


def _preprocess(pil_img: Image.Image) -> np.ndarray:
    img = pil_img.convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE), Image.LANCZOS)
    return np.array(img, dtype=np.float32) / 127.5 - 1.0


def _to_pil(arr: np.ndarray) -> Image.Image:
    return Image.fromarray((np.clip(arr, 0.0, 1.0) * 255).astype(np.uint8))


def generate_sprites(
    input_img: Image.Image,
    diffusion_steps: int,
    num_samples: int,
) -> Image.Image:
    if input_img is None:
        return None

    model = _get_model()

    cond = _preprocess(input_img)
    cond_batch = tf.constant(np.stack([cond] * int(num_samples)))

    generated, _ = model.generate(
        condition_images=cond_batch,
        diffusion_steps=int(diffusion_steps),
    )

    pil_imgs = [_to_pil(generated[i].numpy()) for i in range(int(num_samples))]

    gap = 4
    total_w = sum(img.width for img in pil_imgs) + gap * (len(pil_imgs) - 1)
    strip = Image.new("RGB", (total_w, pil_imgs[0].height), (200, 200, 200))
    x = 0
    for img in pil_imgs:
        strip.paste(img, (x, 0))
        x += img.width + gap

    return strip


with gr.Blocks(title="PathForge") as app:
    gr.Markdown("# PathForge")
    gr.Markdown(
        "Upload a path planning map (gray background, black obstacles, "
        "green start marker, red goal marker) to generate predicted paths."
    )

    with gr.Row():
        input_img = gr.Image(type="pil", label="Condition map (input)")
        output_img = gr.Image(type="pil", label="Generated paths")

    steps_slider = gr.Slider(minimum=10, maximum=200, value=100, step=1, label="Diffusion steps")
    samples_slider = gr.Slider(minimum=1, maximum=8, value=4, step=1, label="Number of samples")
    button = gr.Button("Generate")

    button.click(
        fn=generate_sprites,
        inputs=[input_img, steps_slider, samples_slider],
        outputs=output_img,
    )

if __name__ == "__main__":
    _get_model()
    app.launch(share=True)
