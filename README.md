# Diffusion Model for Path Planning

A dataset generation pipeline for training a diffusion model to plan collision-free paths on 2D grid maps. The pipeline produces paired **condition** / **target** images that can be used as input/output pairs for image-conditioned diffusion training.

## How it works

1. Random occupancy-grid maps are generated with rectangular obstacles.
2. A start and goal cell are sampled such that they are far apart and unobstructed.
3. A\* (8-connected) finds an optimal path between them.
4. The path is perturbed multiple times to create diverse alternatives for the same map — this helps the model learn multimodal distributions.
5. Two 64×64 RGB images are rendered per sample:
   - **Condition** — map + start (green) + goal (red), no path.
   - **Target** — same as condition with the path drawn in blue.
6. All samples and their metadata are saved to `data/`.

## Dataset structure

```
data/
  condition/   # 000000.png … NNNNNN.png  (input images)
  target/      # 000000.png … NNNNNN.png  (output images)
  metadata.json
```

`metadata.json` is a list of objects:

```json
{ "id": 0, "start": [r, c], "goal": [r, c], "path_length": 42 }
```

## Setup

Requires [uv](https://github.com/astral-sh/uv).

```bash
uv sync
```

This creates a `.venv` and installs all dependencies from `uv.lock`.

## Generate data

```bash
uv run python data_generation/generate_data.py
```

Default: **10 000 samples**, 3 paths per map, 64×64 grid. Edit the parameters block at the top of `generate_data.py` to change them.

## Dependencies

| Package | Role |
|---------|------|
| `numpy` | Grid operations and random perturbation |
| `Pillow` | Image rendering |
| `tqdm` | Progress bar |
