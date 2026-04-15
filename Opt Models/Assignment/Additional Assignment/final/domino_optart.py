from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Literal, List, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps


# -------------------------
# Data definitions
# -------------------------


@dataclass(frozen=True, slots=True)
class CellGridSpec:
    rows: int
    cols: int

    def __post_init__(self) -> None:
        if self.rows <= 0 or self.cols <= 0:
            raise ValueError("rows/cols must be positive")


@dataclass(frozen=True, slots=True)
class Domino:
    a: int
    b: int

    def __post_init__(self) -> None:
        if not (0 <= self.a <= 9 and 0 <= self.b <= 9):
            raise ValueError("Domino values must be in 0..9")
        if self.a > self.b:
            raise ValueError("Domino inventory uses a<=b (unordered pair)")

    def orientations(self) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        if self.a == self.b:
            return ((self.a, self.b), (self.a, self.b))
        return ((self.a, self.b), (self.b, self.a))


def standard_domino_set() -> List[Domino]:
    tiles: List[Domino] = []
    for a in range(10):
        for b in range(a, 10):
            tiles.append(Domino(a=a, b=b))
    if len(tiles) != 55:
        raise RuntimeError("Expected 55 dominoes in a double-9 set")
    return tiles


@dataclass(frozen=True, slots=True)
class VerticalSlot:
    r: int
    c: int

    def cells(self) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        return ((self.r, self.c), (self.r + 1, self.c))


def vertical_slots_nonoverlap(rows: int, cols: int) -> List[VerticalSlot]:
    if rows % 2 != 0:
        raise ValueError("rows must be even for non-overlapping vertical tiling")
    return [VerticalSlot(r=r, c=c) for r in range(0, rows, 2) for c in range(cols)]


# -------------------------
# Preprocess
# -------------------------


def quantize_0_to_9(gray_0_255: np.ndarray) -> np.ndarray:
    clipped = np.clip(gray_0_255.astype(np.float32), 0.0, 255.0)
    q = np.rint(clipped / 255.0 * 9.0).astype(np.int64)
    return np.clip(q, 0, 9)


def image_to_cell_grays(
    img: Image.Image,
    grid: CellGridSpec,
    *,
    invert: bool = False,
    resize_mode: Literal["stretch", "center_crop", "letterbox"] = "center_crop",
    letterbox_gray: int = 255,
    center: Tuple[float, float] = (0.5, 0.5),
) -> np.ndarray:
    g = img.convert("L")
    if invert:
        g = Image.fromarray(255 - np.array(g, dtype=np.uint8), mode="L")

    target_wh = (grid.cols, grid.rows)
    if resize_mode == "stretch":
        resized = g.resize(target_wh, resample=Image.Resampling.BOX)
    elif resize_mode == "center_crop":
        cx, cy = center
        cx = float(np.clip(cx, 0.0, 1.0))
        cy = float(np.clip(cy, 0.0, 1.0))
        resized = ImageOps.fit(g, target_wh, method=Image.Resampling.BOX, centering=(cx, cy))
    elif resize_mode == "letterbox":
        canvas = Image.new("L", target_wh, int(np.clip(letterbox_gray, 0, 255)))
        g2 = g.copy()
        g2.thumbnail(target_wh, resample=Image.Resampling.BOX)
        x0 = (target_wh[0] - g2.size[0]) // 2
        y0 = (target_wh[1] - g2.size[1]) // 2
        canvas.paste(g2, (x0, y0))
        resized = canvas
    else:
        raise ValueError("resize_mode must be 'stretch', 'center_crop', or 'letterbox'")

    return np.array(resized, dtype=np.float32)


def map_image_to_grid_image(
    img: Image.Image,
    grid: CellGridSpec,
    *,
    invert: bool = False,
    resize_mode: Literal["stretch", "center_crop", "letterbox"] = "center_crop",
    letterbox_gray: int = 255,
    center: Tuple[float, float] = (0.5, 0.5),
) -> Image.Image:
    g = img.convert("L")
    if invert:
        g = Image.fromarray(255 - np.array(g, dtype=np.uint8), mode="L")
    target_wh = (grid.cols, grid.rows)
    if resize_mode == "stretch":
        return g.resize(target_wh, resample=Image.Resampling.BOX)
    if resize_mode == "center_crop":
        cx, cy = center
        cx = float(np.clip(cx, 0.0, 1.0))
        cy = float(np.clip(cy, 0.0, 1.0))
        return ImageOps.fit(g, target_wh, method=Image.Resampling.BOX, centering=(cx, cy))
    if resize_mode == "letterbox":
        canvas = Image.new("L", target_wh, int(np.clip(letterbox_gray, 0, 255)))
        g2 = g.copy()
        g2.thumbnail(target_wh, resample=Image.Resampling.BOX)
        x0 = (target_wh[0] - g2.size[0]) // 2
        y0 = (target_wh[1] - g2.size[1]) // 2
        canvas.paste(g2, (x0, y0))
        return canvas
    raise ValueError("resize_mode must be 'stretch', 'center_crop', or 'letterbox'")


def make_test_image(*, px_width: int = 1200, px_height: int = 1000, seed: int = 7) -> Image.Image:
    rng = np.random.default_rng(seed)
    img = Image.new("RGB", (px_width, px_height), (245, 245, 245))
    draw = ImageDraw.Draw(img)
    margin = int(min(px_width, px_height) * 0.12)
    bbox = (margin, margin, px_width - margin, px_height - margin)
    draw.ellipse(bbox, outline=(10, 10, 10), width=10, fill=(230, 230, 230))

    eye_r = int(min(px_width, px_height) * 0.05)
    for x in (int(px_width * 0.38), int(px_width * 0.62)):
        y = int(px_height * 0.42)
        draw.ellipse((x - eye_r, y - eye_r, x + eye_r, y + eye_r), fill=(20, 20, 20))

    mouth_bbox = (int(px_width * 0.35), int(px_height * 0.45), int(px_width * 0.65), int(px_height * 0.78))
    draw.arc(mouth_bbox, start=10, end=170, fill=(30, 30, 30), width=12)

    for _ in range(40):
        w = int(rng.integers(20, 80))
        h = int(rng.integers(10, 60))
        x0 = int(rng.integers(0, px_width - w))
        y0 = int(rng.integers(0, px_height - h))
        shade = int(rng.integers(60, 220))
        draw.rectangle((x0, y0, x0 + w, y0 + h), fill=(shade, shade, shade), outline=None)

    try:
        font = ImageFont.truetype("Arial.ttf", size=int(px_height * 0.08))
    except Exception:
        font = ImageFont.load_default()
    draw.text((int(px_width * 0.05), int(px_height * 0.03)), "OPT ART", fill=(0, 0, 0), font=font)
    return img


def save_grid_preview(beta_0_to_9: np.ndarray, *, cell_px: int, out_path: Path) -> None:
    rows, cols = beta_0_to_9.shape
    img = Image.new("L", (cols * cell_px, rows * cell_px), 255)
    draw = ImageDraw.Draw(img)
    for i in range(rows):
        for j in range(cols):
            v = int(beta_0_to_9[i, j])
            shade = int(round(v / 9.0 * 255))
            x0, y0 = j * cell_px, i * cell_px
            draw.rectangle((x0, y0, x0 + cell_px - 1, y0 + cell_px - 1), fill=shade)
    for i in range(rows + 1):
        y = i * cell_px
        draw.line((0, y, cols * cell_px, y), fill=128, width=1)
    for j in range(cols + 1):
        x = j * cell_px
        draw.line((x, 0, x, rows * cell_px), fill=128, width=1)
    img.save(out_path)


# -------------------------
# Costs + solve
# -------------------------


def compute_vertical_cost_matrix(beta_0_to_9: np.ndarray, dominoes: Sequence[Domino], slots: Sequence[VerticalSlot]) -> np.ndarray:
    rows, cols = beta_0_to_9.shape
    beta = beta_0_to_9.astype(np.int64)
    C = np.zeros((len(dominoes), len(slots)), dtype=np.int64)
    for si, slot in enumerate(slots):
        (r0, c0), (r1, c1) = slot.cells()
        if not (0 <= r0 < rows and 0 <= r1 < rows and 0 <= c0 < cols and c0 == c1):
            raise ValueError("Slot out of bounds or malformed")
        top = int(beta[r0, c0])
        bot = int(beta[r1, c1])
        for di, d in enumerate(dominoes):
            (a1, b1), (a2, b2) = d.orientations()
            e1 = (a1 - top) ** 2 + (b1 - bot) ** 2
            e2 = (a2 - top) ** 2 + (b2 - bot) ** 2
            C[di, si] = min(e1, e2)
    return C


@dataclass(frozen=True, slots=True)
class AssignmentSolution:
    objective: float
    slot_to_domino: Dict[int, int]


def solve_domino_assignment(
    C: np.ndarray,
    *,
    method: str,
    solver: str,
    time_limit_s: int | None,
    msg: bool,
    binary: bool = False,
) -> AssignmentSolution:
    if C.ndim != 2:
        raise ValueError("C must be 2D")
    n_d, n_s = C.shape
    if n_d != n_s:
        raise ValueError(f"Need square cost matrix; got {C.shape}")

    if method == "hungarian":
        # Fast exact solver for linear assignment (no MILP build).
        from scipy.optimize import linear_sum_assignment

        row_ind, col_ind = linear_sum_assignment(C)
        slot_to_domino = {int(s): int(d) for d, s in zip(row_ind, col_ind, strict=True)}
        obj = float(C[row_ind, col_ind].sum())
        return AssignmentSolution(objective=obj, slot_to_domino=slot_to_domino)

    if method != "pulp":
        raise ValueError("method must be 'hungarian' or 'pulp'")

    import pulp

    prob = pulp.LpProblem("domino_assignment", pulp.LpMinimize)
    cat = "Binary" if binary else "Continuous"
    x = pulp.LpVariable.dicts("x", (range(n_d), range(n_s)), lowBound=0, upBound=1, cat=cat)
    prob += pulp.lpSum(C[d, s] * x[d][s] for d in range(n_d) for s in range(n_s))
    for s in range(n_s):
        prob += pulp.lpSum(x[d][s] for d in range(n_d)) == 1
    for d in range(n_d):
        prob += pulp.lpSum(x[d][s] for s in range(n_s)) == 1

    if solver == "pulp_cbc":
        pulp_solver: pulp.LpSolver = pulp.PULP_CBC_CMD(msg=msg, timeLimit=time_limit_s)
    elif solver == "gurobi":
        pulp_solver = pulp.GUROBI_CMD(msg=msg, timeLimit=time_limit_s)
    else:
        raise ValueError("solver must be 'pulp_cbc' or 'gurobi'")

    status = prob.solve(pulp_solver)
    status_str = pulp.LpStatus.get(status, str(status))
    if status_str not in {"Optimal", "Feasible"}:
        raise RuntimeError(f"Solver status: {status_str}")

    slot_to_domino: Dict[int, int] = {}
    for s in range(n_s):
        best_d = None
        best_val = -1.0
        for d in range(n_d):
            v = float(pulp.value(x[d][s]) or 0.0)
            if v > best_val:
                best_val = v
                best_d = d
        if best_d is None or best_val < 0.5:
            raise RuntimeError("Failed to extract a valid assignment from solution")
        slot_to_domino[s] = int(best_d)

    return AssignmentSolution(objective=float(pulp.value(prob.objective)), slot_to_domino=slot_to_domino)


# -------------------------
# Render
# -------------------------


def value_to_gray(v: int) -> int:
    v = max(0, min(9, int(v)))
    return int(round(v / 9.0 * 255))


def render_vertical_solution(
    *,
    grid_rows: int,
    grid_cols: int,
    slots: Sequence[VerticalSlot],
    dominoes: Sequence[Domino],
    slot_to_domino: Dict[int, int],
    beta_0_to_9: np.ndarray | None,
    cell_px: int,
    margin_px: int,
) -> Image.Image:
    w = margin_px * 2 + grid_cols * cell_px
    h = margin_px * 2 + grid_rows * cell_px
    img = Image.new("RGB", (w, h), (255, 255, 255))
    draw = ImageDraw.Draw(img)

    if beta_0_to_9 is not None:
        for r in range(grid_rows):
            for c in range(grid_cols):
                shade = value_to_gray(int(beta_0_to_9[r, c]))
                x0 = margin_px + c * cell_px
                y0 = margin_px + r * cell_px
                draw.rectangle((x0, y0, x0 + cell_px - 1, y0 + cell_px - 1), fill=(shade, shade, shade))

    border = max(1, cell_px // 18)
    for si, slot in enumerate(slots):
        if si not in slot_to_domino:
            continue
        d = dominoes[slot_to_domino[si]]
        (r0, c0), (r1, c1) = slot.cells()
        x0 = margin_px + c0 * cell_px
        y0 = margin_px + r0 * cell_px
        x1 = x0 + cell_px
        y1 = y0 + 2 * cell_px
        draw.rectangle((x0, y0, x1, y1), outline=(0, 0, 0), width=border, fill=None)

        top_val, bot_val = d.a, d.b
        if beta_0_to_9 is not None:
            tgt_top = int(beta_0_to_9[r0, c0])
            tgt_bot = int(beta_0_to_9[r1, c1])
            e1 = (d.a - tgt_top) ** 2 + (d.b - tgt_bot) ** 2
            e2 = (d.b - tgt_top) ** 2 + (d.a - tgt_bot) ** 2
            if e2 < e1:
                top_val, bot_val = d.b, d.a

        top_shade = value_to_gray(top_val)
        bot_shade = value_to_gray(bot_val)

        draw.rectangle((x0 + border, y0 + border, x1 - border, y0 + cell_px - border), fill=(top_shade,) * 3)
        draw.rectangle((x0 + border, y0 + cell_px + border, x1 - border, y1 - border), fill=(bot_shade,) * 3)
        draw.line((x0 + border, y0 + cell_px, x1 - border, y0 + cell_px), fill=(0, 0, 0), width=border)

    return img


def main() -> None:
    p = argparse.ArgumentParser(description="Vertical-only domino opt-art (single-file).")
    p.add_argument("--rows", type=int, default=88, help="Grid rows (cells). Must be even.")
    p.add_argument("--cols", type=int, default=50, help="Grid cols (cells).")
    p.add_argument("--image", type=str, default=None, help="Path to input image. If omitted, uses test image.")
    p.add_argument("--resize-mode", choices=["stretch", "center_crop", "letterbox"], default="center_crop")
    p.add_argument("--center-x", type=float, default=0.5, help="Crop center x in [0,1] for center_crop mode.")
    p.add_argument("--center-y", type=float, default=0.5, help="Crop center y in [0,1] for center_crop mode.")
    p.add_argument("--invert", action="store_true")
    p.add_argument(
        "--method",
        choices=["hungarian", "pulp"],
        default="hungarian",
        help="Assignment solve method. Hungarian is fast and exact; PuLP is slower for large grids.",
    )
    p.add_argument("--solver", choices=["pulp_cbc", "gurobi"], default="pulp_cbc")
    p.add_argument("--time-limit-s", type=int, default=None)
    p.add_argument(
        "--auto-grid",
        action="store_true",
        help="Choose rows/cols automatically to match image aspect ratio (avoids crop/letterbox backgrounds).",
    )
    p.add_argument(
        "--auto-base-rows",
        type=int,
        default=88,
        help="Base rows to start from when --auto-grid is set (must be even; will be adjusted if needed).",
    )
    p.add_argument(
        "--auto-target-slots",
        type=int,
        default=2200,
        help="Target number of slots |S| to match (roughly controls detail). Example: 88x50 => |S|=2200.",
    )
    p.add_argument(
        "--preview-only",
        action="store_true",
        help="Only write mapped_input + quantized_grid, skip optimization/render (useful for crop tuning).",
    )
    p.add_argument("--preview-cell-px", type=int, default=18)
    p.add_argument("--render-cell-px", type=int, default=22)
    p.add_argument("--render-margin-px", type=int, default=20)
    p.add_argument("--out-dir", type=str, default="outputs")
    args = p.parse_args()

    if args.rows % 2 != 0:
        raise SystemExit("--rows must be even")

    base_dir = Path(__file__).resolve().parent
    out_dir = (base_dir / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.image is None:
        img = make_test_image()
        src = None
    else:
        src = Path(args.image).expanduser().resolve()
        img = Image.open(src).convert("RGB")

    def choose_grid_for_aspect(image_w: int, image_h: int) -> CellGridSpec:
        # Keep rows even; aim for rows/cols ratio close to image_h/image_w.
        base_rows = args.auto_base_rows
        if base_rows % 2 != 0:
            base_rows += 1
        # initial cols from aspect
        aspect = image_w / max(1, image_h)
        cols0 = max(1, int(round(base_rows * aspect)))
        # Search nearby rows/cols to satisfy:
        # 1) rows even
        # 2) |S|=(rows/2)*cols multiple of 55
        # 3) |S| close to auto_target_slots
        best: tuple[float, int, int] | None = None  # (score, rows, cols)
        for dr in range(-10, 11, 2):
            rows = base_rows + dr
            if rows <= 0 or rows % 2 != 0:
                continue
            cols_guess = max(1, int(round(rows * aspect)))
            for dc in range(-25, 26):
                cols = cols_guess + dc
                if cols <= 0:
                    continue
                slots = (rows // 2) * cols
                if slots % 55 != 0:
                    continue
                # score: prioritize aspect match, then closeness to target slots, then smaller size
                grid_aspect = cols / rows
                aspect_err = abs(grid_aspect - aspect)
                slots_err = abs(slots - args.auto_target_slots) / max(1.0, float(args.auto_target_slots))
                score = aspect_err * 4.0 + slots_err * 1.0 + (rows * cols) / 1e7
                if best is None or score < best[0]:
                    best = (score, rows, cols)
        if best is None:
            raise SystemExit("Failed to auto-select a feasible grid. Try a different --auto-base-rows.")
        _, r, c = best
        return CellGridSpec(rows=r, cols=c)

    if args.auto_grid:
        grid = choose_grid_for_aspect(img.width, img.height)
        # When grid matches aspect, no crop/letterbox is needed; stretch won't distort.
        resize_mode = "stretch"
        center = (0.5, 0.5)
    else:
        grid = CellGridSpec(rows=args.rows, cols=args.cols)
        resize_mode = args.resize_mode
        center = (args.center_x, args.center_y)

    slots = vertical_slots_nonoverlap(grid.rows, grid.cols)
    n_slots = len(slots)
    if n_slots % 55 != 0:
        raise SystemExit(f"|S|={(grid.rows//2)*grid.cols} must be a multiple of 55 (got {n_slots}).")
    N = n_slots // 55

    if src is None:
        img_path = out_dir / f"vertical_input_{grid.rows}x{grid.cols}.png"
    else:
        suffix = src.suffix.lower() or ".png"
        img_path = out_dir / f"vertical_input_{grid.rows}x{grid.cols}{suffix}"
    img.save(img_path)

    mapped = map_image_to_grid_image(img, grid, invert=args.invert, resize_mode=resize_mode, center=center)
    mapped_path = out_dir / f"vertical_mapped_input_{grid.rows}x{grid.cols}.png"
    mapped.save(mapped_path)

    gray = image_to_cell_grays(img, grid, invert=args.invert, resize_mode=resize_mode, center=center)
    beta = quantize_0_to_9(gray)

    q_path = out_dir / f"vertical_quantized_grid_{grid.rows}x{grid.cols}.png"
    save_grid_preview(beta, cell_px=args.preview_cell_px, out_path=q_path)

    if args.preview_only:
        print("Wrote:")
        print(f"- {img_path}")
        print(f"- {mapped_path}")
        print(f"- {q_path}")
        print(f"Grid: {grid.rows}x{grid.cols}  |S|={n_slots}  N={N}")
        print("Preview-only: skipped optimization/render.")
        return

    dominoes = standard_domino_set() * N
    C = compute_vertical_cost_matrix(beta, dominoes, slots)
    sol = solve_domino_assignment(
        C,
        method=args.method,
        solver=args.solver,
        time_limit_s=args.time_limit_s,
        msg=True,
    )

    out = render_vertical_solution(
        grid_rows=grid.rows,
        grid_cols=grid.cols,
        slots=slots,
        dominoes=dominoes,
        slot_to_domino=sol.slot_to_domino,
        beta_0_to_9=beta,
        cell_px=args.render_cell_px,
        margin_px=args.render_margin_px,
    )
    out_path = out_dir / f"vertical_solution_{grid.rows}x{grid.cols}_N{N}.png"
    out.save(out_path)

    print("Wrote:")
    print(f"- {img_path}")
    print(f"- {mapped_path}")
    print(f"- {q_path}")
    print(f"- {out_path}")
    print(f"N sets: {N}")
    print(f"Objective: {sol.objective:.2f}")


if __name__ == "__main__":
    main()

