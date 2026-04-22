import heapq
import json
import os
import random

import numpy as np
from PIL import Image, ImageDraw
from tqdm import tqdm

# =============================================================================
# PARAMETERS
# =============================================================================

NUM_SAMPLES = 10000  # total (map, path) pairs to generate
PATHS_PER_MAP = 3  # multiple paths per map → helps multimodality
MAP_SIZE = 64  # grid is MAP_SIZE x MAP_SIZE
NUM_OBSTACLES = 5  # number of rectangular obstacles per map
OBSTACLE_MIN_W = 4  # obstacle min width  (in grid cells)
OBSTACLE_MAX_W = 16  # obstacle max width
OBSTACLE_MIN_H = 4  # obstacle min height
OBSTACLE_MAX_H = 20  # obstacle max height
MIN_START_GOAL_DIST = 20  # manhattan dist between start and goal
PATH_LINE_WIDTH = 2  # pixel width of drawn path
START_GOAL_RADIUS = 3  # pixel radius of start/goal dot
OUTPUT_DIR = "data"  # where to save everything
SEED = 42

# Colors (RGB)
COLOR_BG = (15, 15, 30)  # dark background
COLOR_OBSTACLE = (80, 80, 120)  # obstacle fill
COLOR_PATH = (96, 165, 250)  # path line   (blue)
COLOR_START = (34, 197, 94)  # start dot   (green)
COLOR_GOAL = (239, 68, 68)  # goal dot    (red)

# =============================================================================

random.seed(SEED)
np.random.seed(SEED)


# ─────────────────────────────────────────────
# Map generation
# ─────────────────────────────────────────────


def generate_map():
    """Return binary occupancy grid: 1 = obstacle, 0 = free."""
    grid = np.zeros((MAP_SIZE, MAP_SIZE), dtype=np.uint8)
    for _ in range(NUM_OBSTACLES):
        w = random.randint(OBSTACLE_MIN_W, OBSTACLE_MAX_W)
        h = random.randint(OBSTACLE_MIN_H, OBSTACLE_MAX_H)
        x = random.randint(0, MAP_SIZE - w - 1)
        y = random.randint(0, MAP_SIZE - h - 1)
        grid[y : y + h, x : x + w] = 1
    return grid


def sample_free_cell(grid):
    """Pick a random free (non-obstacle) cell."""
    free = np.argwhere(grid == 0)
    idx = random.randint(0, len(free) - 1)
    return tuple(free[idx])  # (row, col)


def far_enough(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1]) >= MIN_START_GOAL_DIST


# ─────────────────────────────────────────────
# A* path finding
# ─────────────────────────────────────────────


def heuristic(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def astar(grid, start, goal):
    """
    A* on a 2-D grid with 8-connectivity.
    Returns list of (row, col) from start to goal, or None if no path.
    """
    rows, cols = grid.shape
    open_heap = []
    heapq.heappush(open_heap, (0 + heuristic(start, goal), 0, start))

    came_from = {}
    g_score = {start: 0}
    closed = set()

    directions = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]

    while open_heap:
        _, g, current = heapq.heappop(open_heap)

        if current == goal:
            # reconstruct
            path = []
            node = current
            while node in came_from:
                path.append(node)
                node = came_from[node]
            path.append(start)
            return path[::-1]

        if current in closed:
            continue
        closed.add(current)

        for dr, dc in directions:
            nr, nc = current[0] + dr, current[1] + dc
            if not (0 <= nr < rows and 0 <= nc < cols):
                continue
            if grid[nr, nc] == 1:
                continue
            neighbor = (nr, nc)
            move_cost = 1.414 if (dr != 0 and dc != 0) else 1.0
            tentative = g + move_cost
            if tentative < g_score.get(neighbor, float("inf")):
                came_from[neighbor] = current
                g_score[neighbor] = tentative
                f = tentative + heuristic(neighbor, goal)
                heapq.heappush(open_heap, (f, tentative, neighbor))

    return None  # no path found


def perturb_path(path, grid, noise_std=2.0):
    """
    Return a slightly different valid path by perturbing the midpoint
    and re-running A* from start→mid and mid→goal.
    This creates path diversity for the same (map, start, goal).
    """
    if len(path) < 4:
        return path

    mid_idx = len(path) // 2
    mid_orig = path[mid_idx]

    for _ in range(20):  # try up to 20 random perturbations
        dr = int(np.random.normal(0, noise_std))
        dc = int(np.random.normal(0, noise_std))
        nr = np.clip(mid_orig[0] + dr, 0, MAP_SIZE - 1)
        nc = np.clip(mid_orig[1] + dc, 0, MAP_SIZE - 1)
        mid_new = (int(nr), int(nc))
        if grid[mid_new] == 1:
            continue
        p1 = astar(grid, path[0], mid_new)
        p2 = astar(grid, mid_new, path[-1])
        if p1 and p2:
            return p1 + p2[1:]  # stitch, skip duplicate midpoint

    return path  # fall back to original


# ─────────────────────────────────────────────
# Image rendering
# ─────────────────────────────────────────────


def render_condition(grid, start, goal):
    """
    Condition image: dark background, gray obstacles,
    green start dot, red goal dot.
    Shape: (MAP_SIZE, MAP_SIZE, 3)
    """
    img = Image.new("RGB", (MAP_SIZE, MAP_SIZE), COLOR_BG)
    draw = ImageDraw.Draw(img)

    # obstacles
    for r in range(MAP_SIZE):
        for c in range(MAP_SIZE):
            if grid[r, c] == 1:
                draw.point((c, r), fill=COLOR_OBSTACLE)

    # start and goal circles
    def circle(draw, rc, radius, color):
        r, c = rc
        draw.ellipse([(c - radius, r - radius), (c + radius, r + radius)], fill=color)

    circle(draw, start, START_GOAL_RADIUS, COLOR_START)
    circle(draw, goal, START_GOAL_RADIUS, COLOR_GOAL)

    return np.array(img)


def render_target(grid, start, goal, path):
    """
    Target image: same as condition but with the path drawn on top.
    Shape: (MAP_SIZE, MAP_SIZE, 3)
    """
    img = Image.new("RGB", (MAP_SIZE, MAP_SIZE), COLOR_BG)
    draw = ImageDraw.Draw(img)

    # obstacles
    for r in range(MAP_SIZE):
        for c in range(MAP_SIZE):
            if grid[r, c] == 1:
                draw.point((c, r), fill=COLOR_OBSTACLE)

    # path line  (col, row) for PIL
    if len(path) >= 2:
        pts = [(c, r) for r, c in path]
        draw.line(pts, fill=COLOR_PATH, width=PATH_LINE_WIDTH)

    # start / goal on top of path
    def circle(draw, rc, radius, color):
        r, c = rc
        draw.ellipse([(c - radius, r - radius), (c + radius, r + radius)], fill=color)

    circle(draw, start, START_GOAL_RADIUS, COLOR_START)
    circle(draw, goal, START_GOAL_RADIUS, COLOR_GOAL)

    return np.array(img)


# ─────────────────────────────────────────────
# Main generation loop
# ─────────────────────────────────────────────


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    cond_dir = os.path.join(OUTPUT_DIR, "condition")
    target_dir = os.path.join(OUTPUT_DIR, "target")
    os.makedirs(cond_dir, exist_ok=True)
    os.makedirs(target_dir, exist_ok=True)

    metadata = []  # list of dicts for each sample
    sample_id = 0
    attempts = 0
    maps_needed = (NUM_SAMPLES + PATHS_PER_MAP - 1) // PATHS_PER_MAP

    pbar = tqdm(total=NUM_SAMPLES, desc="Generating samples")

    while sample_id < NUM_SAMPLES:
        attempts += 1
        if attempts > NUM_SAMPLES * 20:
            print("Too many failed attempts — check parameters.")
            break

        # ── generate a new map ──────────────────────────────────────────
        grid = generate_map()

        # ── sample start / goal ─────────────────────────────────────────
        start = sample_free_cell(grid)
        goal = sample_free_cell(grid)
        if not far_enough(start, goal):
            continue

        # ── base A* path ────────────────────────────────────────────────
        base_path = astar(grid, start, goal)
        if base_path is None:
            continue  # map blocks all routes — skip

        # ── generate PATHS_PER_MAP diverse paths for this map ───────────
        paths = [base_path]
        for _ in range(PATHS_PER_MAP - 1):
            alt = perturb_path(base_path, grid, noise_std=4.0)
            paths.append(alt)

        # ── render and save ─────────────────────────────────────────────
        cond_img = render_condition(grid, start, goal)

        for path in paths:
            if sample_id >= NUM_SAMPLES:
                break

            target_img = render_target(grid, start, goal, path)

            fname = f"{sample_id:06d}.png"
            Image.fromarray(cond_img).save(os.path.join(cond_dir, fname))
            Image.fromarray(target_img).save(os.path.join(target_dir, fname))

            metadata.append(
                {
                    "id": sample_id,
                    "start": list(start),
                    "goal": list(goal),
                    "path_length": len(path),
                }
            )

            sample_id += 1
            pbar.update(1)

    pbar.close()

    # save metadata
    meta_path = os.path.join(OUTPUT_DIR, "metadata.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nDone — {sample_id} samples saved to '{OUTPUT_DIR}/'")
    print(f"  {os.path.join(OUTPUT_DIR, 'condition')}  ← input images")
    print(f"  {os.path.join(OUTPUT_DIR, 'target')}     ← output images")
    print(f"  {meta_path}  ← start/goal/path metadata")


if __name__ == "__main__":
    main()
