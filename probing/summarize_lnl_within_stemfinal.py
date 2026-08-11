"""Aggregate + plot the L-vs-NL-within-stem-final probe results.

Reads the per-(arch, run) CSVs written by run_probes_lnl_within_stemfinal.py and
produces:

  1. summary_<split>.csv -- one row per (arch, subset, layer, probe_type) with
     balanced accuracy / selectivity / control / raw accuracy MEAN and STD across
     the runs (the std is the cross-test-set spread).
  2. trajectory_<split>.png -- balanced-accuracy-by-layer trajectories, one panel
     per subset, one line per architecture (linear probe; shaded +/- run std),
     with the chance line (0.5) and the per-subset majority baseline marked.

Usage:
  summarize_lnl_within_stemfinal.py [--results-dir DIR] [--split SPLIT]
                                    [--probe-type TYPE] [--baselines-dir DIR]
  summarize_lnl_within_stemfinal.py (-h | --help)

Options:
  --results-dir DIR    Per-run probe CSVs [default: data/probing/results_lnl_within_stemfinal].
  --split SPLIT        Data split [default: 10L_90NL].
  --probe-type TYPE    Probe drawn in the trajectory plot, linear or mlp;
                       both go in the CSV [default: linear].
  --baselines-dir DIR  Surface n-gram baseline results (run_ngram_baselines.py);
                       overlaid as horizontal lines and appended to the summary
                       CSV. Silently skipped when absent
                       [default: data/probing/results_ngram_baselines_balanced].
"""

import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from probing import MODEL_TYPES
from probing.utils.cli import parse

SUBSETS = ("stemfinal_same", "stemfinal_differ", "all")
SUBSET_TITLES = {
    "stemfinal_same": "shared stem-final\n(alternation hidden)",
    "stemfinal_differ": "differing stem-final\n(alternation visible)",
    "all": "all instances",
}


def layer_key(row):
    """Sortable index: encoder layers first (0..), then decoder layers."""
    base = 0 if row["layer_type"] == "encoder" else 100
    return base + int(row["layer_index"])


def layer_label(row):
    return f"{row['layer_type'][:3]}{row['layer_index']}"


def load_all(results_dir):
    frames = []
    for arch in MODEL_TYPES:
        for f in sorted(glob.glob(os.path.join(results_dir, arch, "*_lnl_within_stemfinal.csv"))):
            df = pd.read_csv(f)
            df["arch"] = arch
            frames.append(df)
    if not frames:
        raise SystemExit(f"No result CSVs found under {results_dir}")
    return pd.concat(frames, ignore_index=True)


def load_baselines(baselines_dir):
    """Load surface n-gram baseline CSVs (run_ngram_baselines.py), or None.

    Aggregated separately from the probe frame: baseline rows have
    layer_type='ngram' (layer_key cannot place them on the encoder/decoder
    axis) and encode view+order in probe_type (e.g. 'ngram2-src', 'lm3-tgt').
    The pseudo-arch subdirectory name ('ngram', 'infinigram') becomes arch.
    """
    frames = []
    for f in sorted(glob.glob(os.path.join(baselines_dir, "*", "*_lnl_within_stemfinal.csv"))):
        df = pd.read_csv(f)
        df["arch"] = os.path.basename(os.path.dirname(f))
        frames.append(df)
    if not frames:
        return None
    df = pd.concat(frames, ignore_index=True)
    metrics = ["balanced_accuracy", "selectivity", "control_balanced_accuracy",
               "raw_accuracy", "majority_baseline"]
    grp = df.groupby(["arch", "subset", "probe_type", "layer_index"], as_index=False)
    summary = grp.agg(
        n_runs=("balanced_accuracy", "size"),
        **{f"{m}_mean": (m, "mean") for m in metrics},
        **{f"{m}_std": (m, "std") for m in metrics},
    )
    summary["lkey"] = 200 + summary["layer_index"]
    summary["layer"] = summary["probe_type"]
    return summary.sort_values(["arch", "subset", "probe_type", "lkey"])


def parse_args():
    return parse(__doc__, choices=dict(probe_type=("linear", "mlp")))


def main():
    args = parse_args()

    df = load_all(args.results_dir)
    df["lkey"] = df.apply(layer_key, axis=1)
    df["layer"] = df.apply(layer_label, axis=1)

    # 1. summary CSV: mean/std over runs
    metrics = ["balanced_accuracy", "selectivity", "control_balanced_accuracy",
               "raw_accuracy", "majority_baseline"]
    grp = df.groupby(["arch", "subset", "lkey", "layer", "probe_type"], as_index=False)
    summary = grp.agg(
        n_runs=("balanced_accuracy", "size"),
        **{f"{m}_mean": (m, "mean") for m in metrics},
        **{f"{m}_std": (m, "std") for m in metrics},
    ).sort_values(["arch", "subset", "probe_type", "lkey"])
    baselines = load_baselines(args.baselines_dir)
    summary_path = os.path.join(args.results_dir, f"summary_{args.split}.csv")
    out = summary if baselines is None else pd.concat([summary, baselines], ignore_index=True)
    out.to_csv(summary_path, index=False)
    print(f"Wrote {summary_path}  ({len(out)} rows)")

    # 2. trajectory plot
    sub = summary[summary["probe_type"] == args.probe_type]
    fig, axes = plt.subplots(1, len(SUBSETS), figsize=(5.2 * len(SUBSETS), 4.6), sharey=True)
    cmap = plt.get_cmap("tab10")
    archs = [a for a in MODEL_TYPES if a in sub["arch"].unique()]

    for ax, subset in zip(axes, SUBSETS):
        s = sub[sub["subset"] == subset]
        order = s.sort_values("lkey")["layer"].drop_duplicates().tolist()
        x = range(len(order))
        for i, arch in enumerate(archs):
            a = s[s["arch"] == arch].set_index("layer").reindex(order)
            y = a["balanced_accuracy_mean"].values
            e = a["balanced_accuracy_std"].fillna(0).values
            ax.plot(x, y, marker="o", color=cmap(i), label=arch, linewidth=1.8)
            ax.fill_between(x, y - e, y + e, color=cmap(i), alpha=0.12)
        ax.axhline(0.5, ls="--", color="grey", lw=1, label="chance (control)")
        base = s["majority_baseline_mean"].mean()
        ax.axhline(base, ls=":", color="black", lw=1, label=f"maj. baseline ({base:.2f})")
        if baselines is not None:
            b = baselines[baselines["subset"] == subset]
            for prefix, ls, color, name in (
                ("ngram", "--", "dimgray", "ngram clf"),
                ("lm", "-.", "darkred", "ngram LM"),
                ("infinigram", (0, (1, 1)), "purple", "inf-gram LM"),
            ):
                bb = b[b["probe_type"].str.startswith(prefix)]
                if bb.empty:
                    continue
                best = bb.loc[bb["balanced_accuracy_mean"].idxmax()]
                ax.axhline(
                    best["balanced_accuracy_mean"], ls=ls, color=color, lw=1.4,
                    label=f"{name} ({best['probe_type']}) {best['balanced_accuracy_mean']:.2f}",
                )
        ax.set_xticks(list(x))
        ax.set_xticklabels(order, rotation=45)
        ax.set_title(SUBSET_TITLES[subset], fontsize=10)
        ax.set_xlabel("layer")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel(f"L/NL balanced accuracy ({args.probe_type} probe)")
    axes[0].set_ylim(0.4, 1.0)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(archs) + 2,
               fontsize=8, frameon=False, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle(
        f"L vs NL decodability by layer — {args.split} (vanilla = fairseq), mean±run-std over 3 test sets",
        y=1.08, fontsize=11,
    )
    fig.tight_layout()
    png_path = os.path.join(args.results_dir, f"trajectory_{args.split}_{args.probe_type}.png")
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    print(f"Wrote {png_path}")


if __name__ == "__main__":
    main()
