"""
ARC-AGI Hypothesis Solver (NumPy) — single file
- propose_hypotheses(train_pairs) takes ONLY the list of training examples
- ranks hypotheses using training pairs
- applies to ONE test input (test[0]) and prints up to 3 candidate outputs

Run:
  python arc_np_solver_one_test.py /path/to/task.json
Optional:
  python arc_np_solver_one_test.py /path/to/task.json --topk 3
  python arc_np_solver_one_test.py /path/to/task.json --test_index 0
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

Array = np.ndarray


# =========================
# Utilities
# =========================
def grids_equal(a: Array, b: Array) -> bool:
    return bool(np.array_equal(a, b))


def most_frequent_color(g: Array) -> int:
    vals, cnts = np.unique(g, return_counts=True)
    return int(vals[np.argmax(cnts)])


def all_equal_grid(g: Array, val: int) -> bool:
    return bool(np.all(g == val))


def bbox_non_bg(g: Array, bg: int) -> Optional[Tuple[int, int, int, int]]:
    coords = np.argwhere(g != bg)
    if coords.size == 0:
        return None
    rmin, cmin = coords.min(axis=0)
    rmax, cmax = coords.max(axis=0)
    return int(rmin), int(rmax), int(cmin), int(cmax)


def crop_to_bbox(g: Array, bg: int) -> Array:
    bb = bbox_non_bg(g, bg)
    if bb is None:
        return g.copy()
    rmin, rmax, cmin, cmax = bb
    return g[rmin : rmax + 1, cmin : cmax + 1].copy()


def dedupe_keep_order(arrs: List[Array]) -> List[Array]:
    uniq: List[Array] = []
    for a in arrs:
        if not any(np.array_equal(a, b) for b in uniq):
            uniq.append(a)
    return uniq


def print_grid(g: Array) -> None:
    for row in g.tolist():
        print(row)


# =========================
# Hypothesis primitives
# =========================
def erase_color(g: Array, erase_val: int = 5, bg: int = 0) -> Array:
    out = g.copy()
    out[g == erase_val] = bg
    return out


def stamp_blocks_from_markers(g: Array, marker: int = 5, stamp: int = 1) -> Array:
    """
    For each marker cell, stamp a 3x3 block of 'stamp' centered on that marker.
    Output is a blank canvas (zeros) with stamps drawn.
    """
    out = np.zeros_like(g)
    R, C = g.shape
    pts = np.argwhere(g == marker)
    for r, c in pts:
        r0, c0 = int(r) - 1, int(c) - 1
        r1, c1 = int(r) + 2, int(c) + 2  # exclusive
        rr0, cc0 = max(0, r0), max(0, c0)
        rr1, cc1 = min(R, r1), min(C, c1)
        out[rr0:rr1, cc0:cc1] = stamp
    return out


def mirror_bottom_into_top(g: Array, bg: int = 0) -> Array:
    """
    If top rows are empty and bottom rows contain content, mirror bottom upward
    by reversing bottom rows to fill the empty top block.
    """
    out = g.copy()
    non_bg_rows = np.where(np.any(g != bg, axis=1))[0]
    if non_bg_rows.size == 0:
        return out
    start = int(non_bg_rows[0])
    if start <= 0:
        return out
    bottom = g[start:, :]
    need = start
    fill = bottom[::-1, :][:need, :]
    out[:need, :] = fill
    return out


def compare_halves_around_separator(
    g: Array, sep_val: int = 1, out_val: int = 3, bg: int = 0
) -> Array:
    """
    Find a separator column (all sep_val if possible).
    Split left/right around it.
    Output = out_val where left == right, else bg.
    """
    cols = np.where(np.all(g == sep_val, axis=0))[0]
    if cols.size == 0:
        counts = np.sum(g == sep_val, axis=0)
        sep_col = int(np.argmax(counts))
    else:
        sep_col = int(cols[0])

    left = g[:, :sep_col]
    right = g[:, sep_col + 1 :]

    if left.shape[1] == 0 or right.shape[1] == 0 or left.shape[1] != right.shape[1]:
        return g.copy()

    eq = left == right
    return np.where(eq, out_val, bg).astype(int)


def infer_single_non_bg_color_from_outputs(train_pairs, bg: int = 0) -> Optional[int]:
    non_bg = set()
    for ex in train_pairs:
        out = np.array(ex["output"], dtype=int)
        for v in np.unique(out):
            v = int(v)
            if v != bg:
                non_bg.add(v)
    if len(non_bg) == 1:
        return next(iter(non_bg))
    return None


def spiral_walls(n: int, wall_color: int = 3, bg: int = 0) -> Array:
    g = np.full((n, n), bg, dtype=int)
    layer = 0
    while layer < n:
        end = n - layer
        start_col = 0 if layer == 0 else layer - 1

        g[layer, start_col:end] = wall_color
        g[layer:end, n - layer - 1] = wall_color
        g[n - layer - 1, layer:end] = wall_color
        for r in range(n - layer - 1, layer + 1, -1):
            g[r, layer] = wall_color

        layer += 2
    return g


# =========================
# Hypothesis container
# =========================
@dataclass(frozen=True)
class Hypothesis:
    name: str
    func: Callable[[Array], Array]
    complexity: int


# =========================
# Solver
# =========================
class ArcHypothesisSolverNP:
    """
    Key requirement: propose_hypotheses() accepts ONLY the list of training pairs.
    """

    def propose_hypotheses(self, train_pairs: List[Dict]) -> List[Hypothesis]:
        first_inp = np.array(train_pairs[0]["input"], dtype=int)
        bg = most_frequent_color(first_inp)

        hyps: List[Hypothesis] = [
            Hypothesis("identity", lambda g: g.copy(), 10),

            # Crops
            Hypothesis("crop_bbox_0", lambda g: crop_to_bbox(g, 0), 2),
            Hypothesis("crop_bbox_bg", lambda g, bg=bg: crop_to_bbox(g, bg), 2),

            # Rotations / flips
            Hypothesis("rot90_ccw", lambda g: np.rot90(g, 1).copy(), 3),
            Hypothesis("rot180",   lambda g: np.rot90(g, 2).copy(), 3),
            Hypothesis("rot270",   lambda g: np.rot90(g, 3).copy(), 3),
            Hypothesis("flip_lr",  lambda g: np.fliplr(g).copy(), 3),
            Hypothesis("flip_ud",  lambda g: np.flipud(g).copy(), 3),

            # Common patterns from your tasks
            Hypothesis("erase_5_to_0", lambda g: erase_color(g, 5, 0), 2),
            Hypothesis("stamp_3x3_from_5_to_1", lambda g: stamp_blocks_from_markers(g, 5, 1), 2),
            Hypothesis("mirror_bottom_into_top_bg", lambda g, bg=bg: mirror_bottom_into_top(g, bg), 4),
            Hypothesis("mirror_bottom_into_top_0",  lambda g: mirror_bottom_into_top(g, 0), 4),
            Hypothesis("compare_halves_sep1_eq_to_3",
                       lambda g: compare_halves_around_separator(g, 1, 3, 0),
                       5),
        ]

        # Conditional spiral (only added if training outputs suggest one non-bg color)
        wall = infer_single_non_bg_color_from_outputs(train_pairs, bg=0)
        if wall is not None:
            def spiral_if_applicable(g: Array, wc=wall) -> Array:
                R, C = g.shape
                if R == C and all_equal_grid(g, 0):
                    return spiral_walls(R, wall_color=wc, bg=0)
                return g.copy()
            hyps.append(Hypothesis("spiral_walls", spiral_if_applicable, 6))

        return hyps

    def score_hypothesis(self, hyp: Hypothesis, train_pairs: List[Dict]) -> Tuple[int, int]:
        """
        Returns (exact_matches, total_cell_matches).
        """
        exact = 0
        cell = 0
        for ex in train_pairs:
            inp = np.array(ex["input"], dtype=int)
            out = np.array(ex["output"], dtype=int)
            pred = hyp.func(inp)

            if pred.shape == out.shape:
                if grids_equal(pred, out):
                    exact += 1
                cell += int(np.sum(pred == out))
            else:
                # mismatched shapes contribute 0 cell matches
                cell += 0

        return exact, cell

    def rank_hypotheses(
        self, hyps: List[Hypothesis], train_pairs: List[Dict], shortlist: int = 50
    ) -> List[Tuple[int, int, int, Hypothesis]]:
        scored: List[Tuple[int, int, int, Hypothesis]] = []
        for h in hyps:
            exact, cell = self.score_hypothesis(h, train_pairs)
            scored.append((exact, cell, -h.complexity, h))
        scored.sort(reverse=True, key=lambda t: (t[0], t[1], t[2]))
        return scored[: max(1, min(shortlist, len(scored)))]

    def predict_one_test_topk(
        self,
        train_pairs: List[Dict],
        test_input: Array,
        topk: int = 3,
        shortlist: int = 50,
    ) -> List[Array]:
        """
        Returns up to 3 unique predictions for ONE test input.
        """
        topk = min(topk, 3)

        hyps = self.propose_hypotheses(train_pairs)
        ranked = self.rank_hypotheses(hyps, train_pairs, shortlist=shortlist)

        preds: List[Array] = []
        for (_, _, _, h) in ranked:
            preds.append(h.func(test_input))
            uniq = dedupe_keep_order(preds)
            if len(uniq) >= topk:
                preds = uniq
                break

        preds = dedupe_keep_order(preds)[:topk]
        if not preds:
            preds = [test_input.copy()]
        return preds


# =========================
# Load JSON
# =========================
def load_task(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# =========================
# Driver
# =========================
def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("task_json", help="Path to ARC task json")
    parser.add_argument("--topk", type=int, default=3, help="Max predictions to return (<=3)")
    parser.add_argument("--test_index", type=int, default=0, help="Which test case to run")
    parser.add_argument("--shortlist", type=int, default=50, help="How many top hypotheses to try")
    args = parser.parse_args()

    task = load_task(args.task_json)

    train_pairs: List[Dict] = task["train"]          # <-- training pairs list from JSON
    test_cases: List[Dict] = task["test"]

    if not test_cases:
        print("No test cases found in JSON.")
        return 1

    t_idx = max(0, min(args.test_index, len(test_cases) - 1))
    test_input = np.array(test_cases[t_idx]["input"], dtype=int)   # <-- one test input (np.ndarray)

    solver = ArcHypothesisSolverNP()
    preds = solver.predict_one_test_topk(
        train_pairs=train_pairs,
        test_input=test_input,
        topk=args.topk,
        shortlist=args.shortlist,
    )

    print(f"\nTask: {args.task_json}")
    print(f"Train pairs: {len(train_pairs)}")
    print(f"Using test[{t_idx}] input shape: {test_input.shape}")
    print(f"Returning {len(preds)} prediction(s) (max 3)\n")

    for i, p in enumerate(preds):
        print(f"--- Prediction {i}  shape={p.shape} ---")
        print_grid(p)
        print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
