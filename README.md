# Diffusion Model for Path Planning

Image-conditioned diffusion model that generates collision-free paths on 2D grid maps. Given a map with a start (green) and goal (red), the model generates the path (blue).

## Quickstart

**1. Clone the repo**
```bash
git clone <repo-url>
```

**2. Enter the directory**
```bash
cd Diffusion_model_for_path_planning
```

**3. Install dependencies** (requires [uv](https://github.com/astral-sh/uv))
```bash
uv sync
```

**4. Train**
```bash
python src/train.py
```

Checkpoints are saved to `src/checkpoints/<run_timestamp>/` every 25 epochs.

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
