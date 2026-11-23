"""Offline inputs, paired comparisons, and export settings for the paper figures."""

import json
from collections import defaultdict
from importlib import import_module
from math import isfinite
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from src.input.loader import load_gold_records, load_inputs

DEV = Path(__file__).resolve().parents[2]
RUNS = {
    "B01": "B01-C01",
    "B02": "B02-C05",
    "B03": "B03-C01",
    "B04": "B04-C01",
    "B06": "B06-C01",
    "B07": "B07-C01",
}
DATASETS = {
    "banking77": "Banking77",
    "clinc150": "CLINC150",
    "anli_r2r3": "ANLI R2/R3",
    "strategyqa": "StrategyQA",
}
JEV = "jev-1.14.0"
GPT = "gpt-6.1-sol"
DEEPSEEK = "deepseek-v4.1-flash"
BACKENDS = {GPT: "GPT-6.1-Sol", DEEPSEEK: "DeepSeek-v4.1-Flash"}
INK = "#172B3B"
MUTED = "#455968"
GRID = "#D7DFE4"
BLUE = "#205888"
ORANGE = "#99521C"
TEAL = "#197268"
PURPLE = "#694C88"
DATASET_COLORS = dict(zip(DATASETS, (BLUE, TEAL, ORANGE, PURPLE)))
BACKEND_COLORS = {GPT: BLUE, DEEPSEEK: ORANGE}


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def run_json(batch, name):
    return read_json(DEV / "runs" / batch / RUNS[batch] / f"{name}.json")


def prediction_index(batch):
    predictions = {}
    for row in run_json(batch, "predictions"):
        key = (row["dataset_id"], row["sample_id"], row["model_id"], row["arm_id"])
        if key in predictions:
            raise ValueError(f"Duplicate prediction in {RUNS[batch]}: {key}")
        predictions[key] = row
    return predictions


def load_role(role):
    """Join inputs and gold by dataset and sample ID, preserving manifest order."""
    allow_holdout = role == "holdout"
    golds = {
        (gold.dataset_id, gold.sample_id): gold
        for gold in load_gold_records(role=role, danger_enable_holdout_for_b07=allow_holdout)
    }
    rows = {dataset: [] for dataset in DATASETS}
    for sample in load_inputs(allowed_role=role, danger_enable_holdout_for_b07=allow_holdout):
        key = (sample.dataset_id, sample.sample_id)
        rows[sample.dataset_id].append((sample, golds.pop(key)))
    if golds:
        raise ValueError(f"Gold records without matching {role} inputs")
    return rows


def has_class_probabilities(prediction, labels):
    probabilities = prediction.get("probabilities") or {}
    return (
        prediction["status"] == "valid"
        and set(probabilities) == set(labels)
        and all(isfinite(p) and 0.0 <= p <= 1.0 for p in probabilities.values())
        and abs(sum(probabilities.values()) - 1.0) <= 1e-6
    )


def archived_jev_labels(dataset, id_labels):
    """CLINC B02/B07 archives have an extra native REJECT output."""
    if dataset == "clinc150":
        return [*id_labels, "REJECT"]
    return id_labels


def group_stratum(gold):
    if gold.dataset_id == "anli_r2r3":
        return gold.analysis_annotations["round"]
    if gold.dataset_id == "clinc150":
        return str(gold.is_oos)
    return ""


def paired_group_draws(by_dataset, *, repeats, seed):
    """Return the macro paired difference and its bootstrap draws.

    Input rows are (group_id, stratum, paired difference). A dataset is split
    by stratum only when every original group fits wholly within one stratum.
    """
    rng = np.random.default_rng(seed)
    observed = []
    bootstraps = []
    for rows in by_dataset.values():
        cluster_diffs = defaultdict(list)
        cluster_strata = defaultdict(set)
        for group, stratum, diff in rows:
            cluster_diffs[group].append(diff)
            cluster_strata[group].add(stratum)
        if not rows:
            raise ValueError("Paired comparison requires data in every dataset")
        compatible = all(len(tags) == 1 for tags in cluster_strata.values())
        strata = defaultdict(list)
        for group in sorted(cluster_diffs):
            tag = next(iter(cluster_strata[group])) if compatible else ""
            strata[tag].append(group)
        diffs = np.zeros(repeats)
        totals = np.zeros(repeats)
        for tag in sorted(strata):
            groups = strata[tag]
            sums = np.array([sum(cluster_diffs[g]) for g in groups])
            sizes = np.array([len(cluster_diffs[g]) for g in groups])
            for start in range(0, repeats, 256):
                stop = min(start + 256, repeats)
                indices = rng.integers(0, len(groups), size=(stop - start, len(groups)))
                diffs[start:stop] += sums[indices].sum(axis=1)
                totals[start:stop] += sizes[indices].sum(axis=1)
        observed.append(sum(diff for _, _, diff in rows) / len(rows))
        bootstraps.append(diffs / totals)
    return float(np.mean(observed)), np.mean(bootstraps, axis=0)


def configure_plotting():
    import matplotlib

    matplotlib.use("Agg")
    matplotlib.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.labelsize": 9.5,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": MUTED,
        "text.color": INK,
        "axes.labelcolor": INK,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "legend.fontsize": 8.5,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "axes.axisbelow": True,
        "svg.hashsalt": "hybrid_cascade-paper",
    })


def save_figure(fig, output_dir, stem):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output_dir) as temp:
        svg = Path(temp) / f"{stem}.svg"
        png = Path(temp) / f"{stem}.png"
        fig.savefig(svg, bbox_inches="tight", metadata={"Date": None})
        fig.savefig(png, dpi=320, bbox_inches="tight")
        svg.replace(output_dir / svg.name)
        png.replace(output_dir / png.name)


def render_paper_figures(output_dir):
    """Rebuild all four figures into separate directories without API calls."""
    renderers = {
        "fig1_system_architecture": "render_architecture",
        "fig2_calibration_risk": "render_calibration",
        "fig3_paired_contrasts": "render_contrasts",
        "fig4_cost_pareto": "render_pareto",
    }
    for folder, script in renderers.items():
        module = import_module(f"figures.{folder}.{script}")
        module.render_figure(Path(output_dir) / folder)
