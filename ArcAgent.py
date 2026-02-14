import numpy as np

from ArcProblem import ArcProblem
from ArcData import ArcData
from ArcSet import ArcSet
from typing import List, Tuple, Callable, Dict, Optional
from dataclasses import dataclass

Array = np.ndarray

# =========================
# Basic utilities
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
def identity(g: Array) -> Array:
    return g.copy()


def crop_bbox_0(g: Array) -> Array:
    return crop_to_bbox(g, bg=0)


def crop_bbox_bg(g: Array, bg: int) -> Array:
    return crop_to_bbox(g, bg=bg)


def rot90_ccw(g: Array) -> Array:
    return np.rot90(g, 1).copy()


def rot180(g: Array) -> Array:
    return np.rot90(g, 2).copy()


def rot270(g: Array) -> Array:
    return np.rot90(g, 3).copy()


def flip_lr(g: Array) -> Array:
    return np.fliplr(g).copy()


def flip_ud(g: Array) -> Array:
    return np.flipud(g).copy()


def erase_color(g: Array, erase_val: int = 5, bg: int = 0) -> Array:
    out = g.copy()
    out[g == erase_val] = bg
    return out


def stamp_blocks_from_markers(g: Array, marker: int = 5, stamp: int = 1) -> Array:
    """
    For every (r,c) with value == marker, stamp a 3x3 block of 'stamp' centered on (r,c).
    Output is a blank (zeros) canvas of same shape with stamps drawn on it.
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

# errored out in Gradescope..changing the logic
def mirror_bottom_into_top(g: Array, bg: int = 0) -> Array:
    out = g.copy()

    non_bg_rows = np.where(np.any(g != bg, axis=1))[0]
    if non_bg_rows.size == 0:
        return out

    start = int(non_bg_rows[0])   # first active row
    if start <= 0:
        return out

    bottom = g[start:, :]
    need = start                  # how many rows to fill at top
    br = bottom.shape[0]

    if br == 0:
        return out

    mirrored = bottom[::-1, :]    # reverse bottom rows

    # Build exactly `need` rows by repeating mirrored rows if necessary
    idx = np.arange(need) % br
    fill = mirrored[idx, :]

    out[:need, :] = fill
    return out



def compare_halves_around_separator(
    g: Array, sep_val: int = 1, out_val: int = 3, bg: int = 0
) -> Array:
    """
    Find a separator column (all sep_val if possible), split left/right.
    Output cell = out_val iff left[r,c] == right[r,c], else bg.
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


def infer_single_non_bg_color_from_outputs(arc_traning_data, bg: int = 0) -> Optional[int]:
    non_bg = set()
    for training_example in arc_traning_data:
        out = np.array(training_example.get_output_data().data(), dtype=int)
        for v in np.unique(out):
            v = int(v)
            if v != bg:
                non_bg.add(v)
    if len(non_bg) == 1:
        return next(iter(non_bg))
    return None


def spiral_walls(n: int, wall_color: int = 3, bg: int = 0) -> Array:
    """
    Deterministic spiral "walls" pattern (works for tasks where input is all zeros square).
    """
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
# I/O helpers
# =========================
def load_task(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def print_grid(arr: Array) -> None:
    for row in arr.tolist():
        print(row)


class ArcAgent:
    def __init__(self):
        """
        You may add additional variables to this init method. Be aware that it gets called only once
        and then the make_predictions method will get called several times.
        """
        pass

    def make_predictions(self, arc_problem: ArcProblem) -> list[np.ndarray]:
        """
        Write the code in this method to solve the incoming ArcProblem.
        Your agent will receive 1 problem at a time.

        You can add up to THREE (3) the predictions to the
        predictions list provided below that you need to
        return at the end of this method.

        In the Autograder, the test data output in the arc problem will be set to None
        so your agent cannot peek at the answer (even on the public problems).

        Also, if you return more than 3 predictions in the list it
        is considered an ERROR and the test will be automatically
        marked as INCORRECT.
        """

        

        # predictions: list[np.ndarray] = list()
        predictions: list[Array] = []

        '''
        The next 2 lines are only an example of how to populate the predictions list.
        This will just be an empty answer the size of the input data;
        delete it before you start adding your own predictions.
        '''
        # output = np.zeros_like(arc_problem.test_set().get_input_data().data())
        # predictions.append(output)
        
        # test_input_data = arc_problem.test_set().get_input_data().data()
        # ys, xs = np.nonzero(test_input_data)

        # ''' If no non-zero cells exist return np array with 0
        # '''
        # if len(xs) == 0:
        #     return predictions

        # ''' Get min and max indicides of rows
        # '''
        # ymin, ymax = ys.min(), ys.max()

        # ''' Get min and max indicides of columns
        # '''
        # xmin, xmax = xs.min(), xs.max()
        
        # ''' Crop using numpy slicing
        # '''
        # cropped = test_input_data[ymin:ymax + 1, xmin:xmax + 1]

        # predictions.append(cropped);

        shortlist = 100
        top_three = 3
     
        hypotheses = self.propose_hypotheses(arc_problem.training_set())
        ranked = self.rank_hypotheses(hypotheses, arc_problem.training_set(), shortlist=shortlist)

        if not ranked:
            return predictions

        for (_, _, _, h) in ranked:
            predictions.append(h.func(arc_problem.test_set().get_input_data().data()))
            unique_predictions = dedupe_keep_order(predictions)
            if len(unique_predictions) >= top_three:
                predictions = unique_predictions
                break

        predictions = dedupe_keep_order(predictions)[:top_three]
        
        return predictions
 
    
    def propose_hypotheses(self, arc_training_data: ArcProblem) -> List[Hypothesis]:
        #train = training_pairs
        first_tranining_input = arc_training_data[0].get_input_data().data()
        bg = most_frequent_color(first_tranining_input)

        hypotheses  : List[Hypothesis] = [
            Hypothesis("identity", identity, 10),
            Hypothesis("crop_bbox_0", crop_bbox_0, 2),
            Hypothesis("crop_bbox_bg", lambda g, bg=bg: crop_bbox_bg(g, bg), 2),
            Hypothesis("rot90_ccw", rot90_ccw, 3),
            Hypothesis("rot180", rot180, 3),
            Hypothesis("rot270", rot270, 3),
            Hypothesis("flip_lr", flip_lr, 3),
            Hypothesis("flip_ud", flip_ud, 3),
            Hypothesis("erase_5_to_0", lambda g: erase_color(g, 5, 0), 2),
            Hypothesis("stamp_3x3_from_5_to_1", lambda g: stamp_blocks_from_markers(g, 5, 1), 2),
            Hypothesis("mirror_bottom_into_top_bg", lambda g, bg=bg: mirror_bottom_into_top(g, bg), 4),
            Hypothesis("mirror_bottom_into_top_0", lambda g: mirror_bottom_into_top(g, 0), 4),
            Hypothesis("compare_halves_sep1_eq_to_3", lambda g: compare_halves_around_separator(g, 1, 3, 0), 5),
        ]

        # Spiral conditional hypothesis (only meaningful on square all-zero inputs)
        wall = infer_single_non_bg_color_from_outputs(arc_training_data, bg=0)
        if wall is not None:
            def spiral_if_applicable(g: Array, wc=wall) -> Array:
                R, C = g.shape
                if R == C and all_equal_grid(g, 0):
                    return spiral_walls(R, wall_color=wc, bg=0)
                return g.copy()
            hypotheses.append(Hypothesis("spiral_walls", spiral_if_applicable, 6))

        return hypotheses 

    def score_hypothesis(self, hypothesis: Hypothesis, arc_training_data: ArcProblem) -> Tuple[int, int]:
        """
        (exact_matches, total_cell_matches)
        """
        exact = 0
        cell = 0
        for ex in arc_training_data:

            input =  ex.get_input_data().data()
            output = ex.get_output_data().data()

            prediction = hypothesis.func(input)

            if prediction.shape == output.shape:
                if grids_equal(prediction, output):
                    exact += 1
                cell += int(np.sum(prediction == output))
        return exact, cell

    def rank_hypotheses(self, hypotheses: List[Hypothesis], arc_traning_data : ArcProblem, shortlist: int = 50):
        scored = []

        for hypothesis in hypotheses:
            exact, cell = self.score_hypothesis(hypothesis, arc_traning_data)

            # Only keep hypotheses that match at least 1 training example
            if exact > 0:
                scored.append((exact, cell, -hypothesis.complexity, hypothesis))

        scored.sort(reverse=True, key=lambda t: (t[0], t[1], t[2]))
        return scored[: max(1, min(shortlist, len(scored)))]

    # def solve_task_top3(self, task: Dict, max_per_test: int = 3, shortlist: int = 50) -> List[List[np.ndarray]]:
    #     """
    #     Returns: List[test_case][<=3 predictions as np.ndarray]
    #     """
    #     max_per_test = min(max_per_test, 3)

    #     hyps = self.propose_hypotheses(task)
    #     ranked = self.rank_hypotheses(hyps, task["train"], shortlist=shortlist)

    #     results: List[List[np.ndarray]] = []
    #     for ex in task["test"]:
    #         inp = np.array(ex["input"], dtype=int)

    #         preds: List[Array] = []
    #         for (_, _, _, h) in ranked:
    #             preds.append(h.func(inp))
    #             unique_predictions = dedupe_keep_order(preds)
    #             if len(unique_predictions) >= max_per_test:
    #                 preds = unique_predictions
    #                 break

    #         preds = dedupe_keep_order(preds)[:max_per_test]
    #         if not preds:
    #             preds = [inp.copy()]
    #         results.append(preds)

    #     return results