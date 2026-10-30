"""Draw the README figure from saved benchmark scores (no cross-validation is re-run).

Input:  results/benchmark/metrics.csv, written by ``python -m smalldata.run --preset benchmark``
Output: docs/r2_by_protocol_light.png and docs/r2_by_protocol_dark.png

Usage:  python scripts/make_figures.py [path/to/metrics.csv]
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "benchmark" / "metrics.csv"
DOCS = ROOT / "docs"

TARGETS = {
    "passenger_volume_cuft": "Passenger volume",
    "cargo_volume_cuft": "Cargo volume",
    "towing_capacity_lb": "Towing capacity (n = 20)",
    "fuel_efficiency_mpg": "Fuel efficiency",
    "acceleration_0_60_s": "0–60 mph time",
}
MODELS = ["Linear regression", "Lasso", "kNN", "SVR", "Mean", "Ridge (all features)", "Lasso (no SFS)", "Complete procedure"]
FLAT = "flat (thesis re-implementation)"
NESTED = "nested, GroupKFold"

THEMES = {
    "light": {"bg": "#FFFFFF", "text": "#142E46", "grid": "#E3E8EE", "flat": "#4F6F94", "nested": "#087E85"},
    "dark": {"bg": "#0D1117", "text": "#E6EDF3", "grid": "#2A313C", "flat": "#8FA8C8", "nested": "#43BEC4"},
}


def summarise(runs: pd.DataFrame) -> pd.DataFrame:
    runs = runs[runs["protocol"].isin([FLAT, NESTED])]
    return runs.groupby(["protocol", "target", "model"])["R2"].agg(["mean", "std"])


def draw(stats: pd.DataFrame, theme: dict, path: Path) -> None:
    plt.rcParams.update({"font.size": 10, "axes.titlesize": 11, "font.family": "DejaVu Sans"})
    fig, axes = plt.subplots(1, len(TARGETS), figsize=(12, 4.4), sharey=True)
    fig.patch.set_facecolor(theme["bg"])
    rows = range(len(MODELS))[::-1]  # first model on top

    for ax, (target, title) in zip(axes, TARGETS.items()):
        ax.set_facecolor(theme["bg"])
        flat = stats.loc[FLAT].loc[target].reindex(MODELS)  # no flat run exists for the complete procedure
        nested = stats.loc[NESTED].loc[target].reindex(MODELS)
        for y, model in zip(rows, MODELS):
            if pd.notna(flat.at[model, "mean"]):
                ax.plot([nested.at[model, "mean"], flat.at[model, "mean"]], [y, y], color=theme["grid"], lw=3, zorder=1)
        ax.errorbar(flat["mean"], rows, xerr=flat["std"], fmt="o", ms=6, mfc=theme["bg"], mec=theme["flat"],
                    mew=1.6, ecolor=theme["flat"], elinewidth=1, capsize=2, zorder=3, label="flat (thesis protocol, optimistic)")
        ax.errorbar(nested["mean"], rows, xerr=nested["std"], fmt="o", ms=6, color=theme["nested"],
                    elinewidth=1, capsize=2, zorder=4, label="nested GroupKFold")
        ax.axhline(len(MODELS) - 4.5, color=theme["grid"], lw=0.8, ls=":", zorder=0)  # families above, baselines below
        ax.axvline(0, color=theme["text"], lw=0.8, alpha=0.5, zorder=0)
        ax.set_title(title, color=theme["text"])
        ax.set_xlabel("R² (out of fold)", color=theme["text"])
        ax.grid(axis="x", color=theme["grid"], lw=0.8)
        ax.set_axisbelow(True)
        ax.tick_params(colors=theme["text"], length=0)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(theme["grid"])
        lo = min(-0.2, (nested["mean"] - nested["std"]).min() - 0.1)
        ax.set_xlim(max(lo, -2.5), 1.0)  # very poor scores would be cut off at -2.5

    axes[0].set_yticks(list(rows), MODELS)
    handles, labels = axes[0].get_legend_handles_labels()
    leg = fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.02))
    for text in leg.get_texts():
        text.set_color(theme["text"])
    fig.text(0.5, -0.04, "Out-of-fold R², mean ± SD over 5 seeds (each seed reshuffles folds and search draws; not a confidence interval). "
             "Below 0 is worse than the training-fold mean.", ha="center", color=theme["text"], fontsize=9, alpha=0.8)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(path, dpi=150, facecolor=theme["bg"], bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    results = Path(sys.argv[1]) if len(sys.argv) > 1 else RESULTS
    stats = summarise(pd.read_csv(results))
    DOCS.mkdir(exist_ok=True)
    for name, theme in THEMES.items():
        out = DOCS / f"r2_by_protocol_{name}.png"
        draw(stats, theme, out)
        print(f"wrote {out.relative_to(ROOT)} ({out.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
