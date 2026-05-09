# Diffusion Model for Path Planning

An image-conditioned diffusion model that generates collision-free paths on 2D grid maps. Given a map with a start marker (green) and a goal marker (red), the model learns to generate the connecting path (blue) while avoiding obstacles.

---

## Installation & Usage

**1. Clone the repo**
```bash
git clone git@github.com:saifahmadgit/Diffusion_model_for_path_planning.git
```

**2. Enter the directory**
```bash
cd Diffusion_model_for_path_planning
```

**3. Install dependencies** (requires [uv](https://github.com/astral-sh/uv))
```bash
uv sync
source .venv/bin/activate
```

**4. Train**
```bash
python src/train.py
```

Checkpoints are saved to `src/checkpoints/<run_timestamp>/` every 25 epochs. If you are on a machine with limited GPU memory, reduce the batch size:

```bash
python src/train.py --batch-size 8
```

**5. Generate**
```bash
python src/generate.py --checkpoint <run_timestamp>
```

Omit `--checkpoint` to list available runs. Use `--num-images N` to control how many samples are generated (default: 10). Output is saved to `src/samples/<timestamp>/` as side-by-side `comparison_000.png … comparison_NNN.png` (Condition | Generated | Ground Truth).

**6. Regenerate dataset**
```bash
python data_generation/generate_data.py
```

Edit the parameters block at the top of `generate_data.py` to change grid size, number of samples, obstacle counts, etc. Output goes to `data_generation/data/`.

---

## Results

The model was trained for 75 epochs. The results below show the condition map (input), the generated path, and the ground truth side by side.

In these two examples, the model generates a valid path from the green start to the red goal while avoiding all obstacles, though the exact route differs from the ground truth:

![comparison_004](src/samples/75_epochs/comparison_004.png)

![comparison_007](src/samples/75_epochs/comparison_007.png)

In this example, the generated path closely matches the ground truth:

![comparison_009](src/samples/75_epochs/comparison_009.png)

Overall, the model has learned the task: given a map, it reliably generates a path that connects the start and goal while avoiding obstacles.

---

## Extra: Web-based GUI

A Gradio web interface is included for interactive inference.

```bash
python src/app.py
```

The app loads the checkpoint automatically. Once running, it prints two URLs:

```
Running on local URL:  http://127.0.0.1:7861
Running on public URL: https://xxxxxxxxxxxx.gradio.live
```

- **Local** (`127.0.0.1`): only works if you are on the same machine.
- **Public** (`gradio.live`): works from any browser, including your laptop. Use this when running on a remote server over SSH.

Upload a condition map (gray background, black obstacles, green start block, red goal block), adjust the diffusion steps and number of samples, and click **Generate**. The results are displayed side by side in the browser.

> **Note:** On shared servers, `/tmp/gradio/` may have permission issues. The app works around this automatically by using `~/.gradio_tmp` instead.

---

## Difficulties & Solutions

Initially, the model failed to generate valid paths whenever the start and goal markers were far apart. It would only produce reasonable outputs when the two markers were nearly adjacent. The root cause was that standard convolutions capture only local spatial relationships, so the model struggled to relate the green and red pixels when they were distant from each other.

To address this, attention layers were added at the bottleneck and at the encoder and decoder layers immediately surrounding it. Attention allows the model to learn the relative importance of any two spatial locations regardless of distance, enabling it to connect far-apart start and goal positions. Attention was intentionally not added to every layer, since that would significantly increase computational cost. Adding it selectively at the bottleneck and the adjacent encoder/decoder layers was sufficient to resolve the issue.
