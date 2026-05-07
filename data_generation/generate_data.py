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

NUM_SAMPLES = 50000  # total (map, path) pairs to generate
MAP_SIZE = 128  # grid is MAP_SIZE x MAP_SIZE
NUM_OBSTACLES_MIN = 5  # min number of rectangular obstacles per map
NUM_OBSTACLES_MAX = 7  # max number of rectangular obstacles per map
OBSTACLE_MIN_W = 16  # obstacle min width  (in grid cells)
OBSTACLE_MAX_W = 32  # obstacle max width
OBSTACLE_MIN_H = 16  # obstacle min height
OBSTACLE_MAX_H = 32  # obstacle max height
OBSTACLE_MIN_CENTER_DIST = 40  # min distance between obstacle centers
MIN_START_GOAL_DIST = 50  # manhattan dist between start and goal
PATH_LINE_WIDTH = 12  # pixel width of drawn path
MARKER_SIZE = 12      # pixel size of start/goal markers (square)
ROBOT_RADIUS = PATH_LINE_WIDTH // 2  # obstacle inflation radius for C-space planning
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
SEED = 42

# Colors (RGB)
COLOR_BG = (128, 128, 128)  # gray background
COLOR_OBSTACLE = (0, 0, 0)  # black obstacles
COLOR_PATH = (0, 0, 255)  # path line   (blue)
COLOR_START = (34, 197, 94)  # start dot   (green)
COLOR_GOAL = (239, 68, 68)  # goal dot    (red)

# =============================================================================

random.seed(SEED)
np.random.seed(SEED)


# ─────────────────────────────────────────────
# Map generation
# ─────────────────────────────────────────────


def generate_map():
    """Return (grid, num_obstacles). Grid: 1 = obstacle, 0 = free."""
    grid = np.zeros((MAP_SIZE, MAP_SIZE), dtype=np.uint8)
    num_obstacles = random.randint(NUM_OBSTACLES_MIN, NUM_OBSTACLES_MAX)
    centers = []
    placed = 0
    attempts = 0
    while placed < num_obstacles and attempts < 200:
        attempts += 1
        w = random.randint(OBSTACLE_MIN_W, OBSTACLE_MAX_W)
        h = random.randint(OBSTACLE_MIN_H, OBSTACLE_MAX_H)
        x = random.randint(0, MAP_SIZE - w - 1)
        y = random.randint(0, MAP_SIZE - h - 1)
        cx, cy = x + w // 2, y + h // 2
        # reject if center is too close to any existing obstacle center
        if any(
            abs(cx - px) + abs(cy - py) < OBSTACLE_MIN_CENTER_DIST for px, py in centers
        ):
            continue
        grid[y : y + h, x : x + w] = 1
        centers.append((cx, cy))
        placed += 1
    return grid, placed


def inflate_obstacles(grid, radius):
    """Expand every obstacle outward by radius cells (C-space expansion)."""
    padded = np.pad(grid, radius, mode="constant", constant_values=0)
    from numpy.lib.stride_tricks import sliding_window_view
    windows = sliding_window_view(padded, (2 * radius + 1, 2 * radius + 1))
    return windows.max(axis=(-2, -1)).astype(np.uint8)


def sample_free_cell(grid):
    """Pick a random free (non-obstacle) cell with enough margin for the marker."""
    margin = MARKER_SIZE - 1
    free = np.argwhere(
        (grid == 0) &
        (np.arange(MAP_SIZE)[:, None] <= MAP_SIZE - 1 - margin) &
        (np.arange(MAP_SIZE)[None, :] <= MAP_SIZE - 1 - margin)
    )
    idx = random.randint(0, len(free) - 1)
    return tuple(free[idx])  # (row, col)


def far_enough(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1]) >= MIN_START_GOAL_DIST


# ─────────────────────────────────────────────
# A* path finding
# ─────────────────────────────────────────────


def heuristic(a, b):
    dr = abs(a[0] - b[0])
    dc = abs(a[1] - b[1])
    # octile distance: admissible heuristic for 8-directional movement
    return max(dr, dc) + (1.414 - 1.0) * min(dr, dc)


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


def perturb_path(path, inflated_grid, noise_std=2.0):
    grid = inflated_grid
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
    Condition image: gray background, black obstacles,
    green start pixel, red goal pixel.
    Shape: (MAP_SIZE, MAP_SIZE, 3)
    """
    img = Image.new("RGB", (MAP_SIZE, MAP_SIZE), COLOR_BG)
    draw = ImageDraw.Draw(img)

    sr, sc = start
    gr, gc = goal

    for r in range(MAP_SIZE):
        for c in range(MAP_SIZE):
            if grid[r, c] == 1:
                if sr <= r <= sr + MARKER_SIZE - 1 and sc <= c <= sc + MARKER_SIZE - 1:
                    continue
                if gr <= r <= gr + MARKER_SIZE - 1 and gc <= c <= gc + MARKER_SIZE - 1:
                    continue
                draw.point((c, r), fill=COLOR_OBSTACLE)

    def draw_marker(rc, color):
        r, c = rc
        draw.rectangle([(c, r), (c + MARKER_SIZE - 1, r + MARKER_SIZE - 1)], fill=color)

    draw_marker(start, COLOR_START)
    draw_marker(goal, COLOR_GOAL)

    return np.array(img)


def render_target(grid, start, goal, path):
    """
    Target image: same as condition but with the A* path drawn on top.
    Shape: (MAP_SIZE, MAP_SIZE, 3)
    """
    img = Image.new("RGB", (MAP_SIZE, MAP_SIZE), COLOR_BG)
    draw = ImageDraw.Draw(img)

    sr, sc = start
    gr, gc = goal

    for r in range(MAP_SIZE):
        for c in range(MAP_SIZE):
            if grid[r, c] == 1:
                if sr <= r <= sr + 7 and sc <= c <= sc + 7:
                    continue
                if gr <= r <= gr + 7 and gc <= c <= gc + 7:
                    continue
                draw.point((c, r), fill=COLOR_OBSTACLE)

    # path line — (col, row) for PIL
    if len(path) >= 2:
        pts = [(c, r) for r, c in path]
        draw.line(pts, fill=COLOR_PATH, width=PATH_LINE_WIDTH)

    # 8x8 block for visibility; A* uses the exact pixel
    def draw_marker(rc, color):
        r, c = rc
        draw.rectangle([(c, r), (c + 7, r + 7)], fill=color)

    draw_marker(start, COLOR_START)
    draw_marker(goal, COLOR_GOAL)

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

    pbar = tqdm(total=NUM_SAMPLES, desc="Generating samples")

    while sample_id < NUM_SAMPLES:
        attempts += 1
        if attempts > NUM_SAMPLES * 50:
            print("Too many failed attempts — check parameters.")
            break

        # ── generate a new map ──────────────────────────────────────────
        grid, num_obstacles = generate_map()
        inflated = inflate_obstacles(grid, ROBOT_RADIUS)

        # ── sample start / goal (must be free in inflated space) ────────
        start = sample_free_cell(inflated)
        goal = sample_free_cell(inflated)
        if not far_enough(start, goal):
            continue

        # ── A* path on inflated grid so robot width never clips obstacles
        path = astar(inflated, start, goal)
        if path is None:
            continue  # no valid path exists — skip this map

        # ── render on original grid ─────────────────────────────────────
        cond_img = render_condition(grid, start, goal)
        target_img = render_target(grid, start, goal, path)

        fname = f"{sample_id:06d}.png"
        Image.fromarray(cond_img).save(os.path.join(cond_dir, fname))
        Image.fromarray(target_img).save(os.path.join(target_dir, fname))

        metadata.append(
            {
                "id": sample_id,
                "start": [int(x) for x in start],
                "goal": [int(x) for x in goal],
                "path_length": len(path),
                "num_obstacles": num_obstacles,
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
