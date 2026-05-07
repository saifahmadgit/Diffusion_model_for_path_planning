# Diffusion Model for Path Planning

Image-conditioned diffusion model that generates collision-free paths on 2D grid maps. Given a map with a start (green) and goal (red), the model generates the path (blue).

## Setup

Requires [uv](https://github.com/astral-sh/uv).

```bash
git clone <repo-url>
cd Diffusion_model_for_path_planning
uv sync
```

## Train

```bash
cd src
python train.py
```

Checkpoints are saved to `src/checkpoints/<run_timestamp>/` every 25 epochs and at the end of training.

## Generate

```bash
cd src
python generate.py -c <run_timestamp>
```

Omit `-c` to list available checkpoints.

**Options**

| Flag | Short | Default | Description |
|------|-------|---------|-------------|
| `--checkpoint` | `-c` | — | Run folder name (e.g. `20260505_143021`) or absolute path to a `.h5` file |
| `--num-images` | `-n` | `10` | Number of images to generate |

Output is saved to `src/samples/<timestamp>/` as `comparison_000.png … comparison_NNN.png`, each showing **Condition → Generated → Ground Truth** side by side.

## Data generation

To regenerate the dataset:

```bash
python data_generation/generate_data.py
```

Edit the parameters block at the top of `generate_data.py` to change grid size, number of samples, obstacle counts, etc. Output goes to `data_generation/data/`.
