"""Generate every quantitative asset in the paper from the result CSVs.

Prints the paper's table values and writes the boxplot figures; pure
aggregation, no probing. Probe balanced accuracy is aggregated from the
per-run CSVs, not from summary_10L_90NL.csv (whose accuracy_mean is raw).

Usage:
  make_paper_assets.py [--no-figures]
  make_paper_assets.py (-h | --help)

Options:
  --no-figures  Print the tables only; skip figure and CSV regeneration.
"""

import glob
import os

import numpy as np
import pandas as pd

ARCHS = ["vanilla", "character_separated", "feature_invariant",
         "independent_feature", "feature_geometric"]
LABELS = {"vanilla": "Vanilla", "character_separated": "Char-separated",
          "feature_invariant": "Feat-invariant",
          "independent_feature": "Feat-onehot",
          "feature_geometric": "Feat-geometric"}
COLORS = {"vanilla": "#E69F00", "character_separated": "#56B4E9",
          "feature_invariant": "#009E73", "independent_feature": "#0072B2",
          "feature_geometric": "#CC79A7"}
RUNS = ["1_1", "1_2", "1_3", "1_4", "2_1", "2_2", "2_3", "2_4",
        "3_1", "3_2", "3_3", "3_4"]
LAY = ["enc0", "enc1", "enc2", "enc3", "dec0", "dec1", "dec2", "dec3"]
PROPS = ["l_shaped", "conjugation", "stem_final_match"]


def load_per_run(pattern, arch_dirs=ARCHS):
    frames = []
    for a in arch_dirs:
        for f in glob.glob(pattern.format(arch=a)):
            d = pd.read_csv(f)
            d["arch"] = a
            frames.append(d)
    df = pd.concat(frames, ignore_index=True)
    df["layer"] = df.layer_type.str[:3] + df.layer_index.astype(str)
    return df


def best_rows(df, value="balanced_accuracy", by=("arch",)):
    g = df.groupby([*by, "layer"]).agg(
        m=(value, "mean"), sd=(value, "std"),
        sel=("selectivity", "mean"), n=(value, "size")).reset_index()
    out = []
    for key, sub in g.groupby(list(by)):
        s = sub.set_index("layer")
        bi = s.m.idxmax()
        out.append((key if isinstance(key, tuple) else (key,), bi, s.loc[bi]))
    return out


def table_main():
    print("\n=== Main probes (linear, balanced, per-run aggregation) ===")
    df = load_per_run("data/probing/results_stemfinal_lnl_grouped/{arch}/*_results.control.csv")
    df = df[df.probe_type == "linear"]
    for p in PROPS:
        for key, layer, r in best_rows(df[df.property == p]):
            print(f"  {p:<18} {LABELS[key[0]]:<15} {layer} "
                  f"{r.m:.2f}±{r.sd:.2f} sel={r.sel:.2f} (n={int(r.n)})")


def table_subset():
    print("\n=== L/NL by subset (within summary, balanced natively) ===")
    s = pd.read_csv("data/probing/results_lnl_within_stemfinal/summary_10L_90NL.csv")
    for sub in ["all", "stemfinal_differ", "stemfinal_same"]:
        bl = s[(s.arch == "ngram") & (s.subset == sub)].balanced_accuracy_mean.max()
        print(f"  [{sub}] best surface {bl:.2f}")
        for a in ARCHS:
            d = s[(s.arch == a) & (s.subset == sub) & (s.probe_type == "linear")].set_index("layer")
            bi = d.balanced_accuracy_mean.idxmax()
            b = d.loc[bi]
            print(f"    {LABELS[a]:<15} {bi} {b.balanced_accuracy_mean:.2f}"
                  f"±{b.balanced_accuracy_std:.2f} sel={b.selectivity_mean:.2f}")


def table_positional():
    print("\n=== Positional readout (best layer) ===")
    df = load_per_run("data/probing/results_positional/{arch}/*_positional.csv")
    df = df[df.probe_type == "linear"]
    cells = [("stem_final_match", "all"), ("l_shaped", "all"),
             ("conjugation", "all"), ("l_shaped", "stemfinal_same")]
    for p, sub in cells:
        d = df[(df.property == p) & (df.subset == sub)]
        for key, layer, r in best_rows(d):
            print(f"  {p}/{sub:<15} {LABELS[key[0]]:<15} {layer} "
                  f"{r.m:.2f}±{r.sd:.2f} sel={r.sel:+.2f}")


def table_prealt():
    print("\n=== Pre-alternant readout (decoder best layer, predictive state) ===")
    df = load_per_run("data/probing/results_positional/{arch}/*_prealt.csv")
    df = df[df.probe_type == "linear"]
    for p, sub in [("l_shaped", "all"), ("l_shaped", "stemfinal_same"),
                   ("stem_final_match", "all")]:
        d = df[(df.property == p) & (df.subset == sub)]
        for key, layer, r in best_rows(d):
            print(f"  {p}/{sub:<15} {LABELS[key[0]]:<15} {layer} "
                  f"{r.m:.2f}\u00b1{r.sd:.2f} sel={r.sel:+.2f}")


def table_transfer():
    print("\n=== Transfer (best layer per direction) ===")
    df = load_per_run("data/probing/results_transfer/{arch}/*_transfer.csv")
    for direc in ["differ->same", "same->differ"]:
        print(f"  [{direc}]")
        for key, layer, r in best_rows(df[df.direction == direc]):
            print(f"    {LABELS[key[0]]:<15} {layer} {r.m:.3f}±{r.sd:.3f} sel={r.sel:+.3f}")


def boxplot_figure():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs("paper/figures", exist_ok=True)
    df = load_per_run("data/probing/results_stemfinal_lnl_grouped/{arch}/*_results.control.csv")
    df = df[df.probe_type == "linear"]
    bl = pd.concat([pd.read_csv(f) for f in glob.glob(
        "data/probing/results_ngram_baselines_balanced/ngram/*_stemfinal_lnl_results.csv")])
    base = {p: bl[bl.property == p].groupby(
        ["probe_type", "layer_index"]).balanced_accuracy.mean().max() for p in PROPS}
    ws = pd.read_csv("data/probing/results_lnl_within_stemfinal/summary_10L_90NL.csv")
    wsb = ws[(ws.arch == "ngram") & (ws.subset == "all")]
    base["l_shaped"] = max(base["l_shaped"], wsb.balanced_accuracy_mean.max())
    plt.rcParams.update({"font.size": 8.5, "axes.titlesize": 9,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.alpha": 0.25, "pdf.fonttype": 42})
    titles = {"l_shaped": "L-shaped (L vs. NL)",
              "conjugation": "conjugation (-ar/-er/-ir)",
              "stem_final_match": "stem-final match"}
    chance = {"l_shaped": 0.5, "conjugation": 1 / 3, "stem_final_match": 0.5}
    rng = np.random.RandomState(0)
    fig, axes = plt.subplots(1, 3, figsize=(9.8, 2.9))
    for ax, p in zip(axes, ["l_shaped", "conjugation", "stem_final_match"]):
        data, labels, colors, best_layers = [], [], [], []
        short = {"vanilla": "Van", "character_separated": "C-Sep",
                 "feature_invariant": "F-Inv", "independent_feature": "F-1H",
                 "feature_geometric": "F-Geo"}
        for a in ARCHS:
            sub = df[(df.arch == a) & (df.property == p)]
            bl_layer = sub.groupby("layer").balanced_accuracy.mean().idxmax()
            data.append(sub[sub.layer == bl_layer].balanced_accuracy.values)
            labels.append(short[a])
            colors.append(COLORS[a])
            best_layers.append(bl_layer)
        bp = ax.boxplot(data, positions=range(5), widths=0.55, patch_artist=True,
                        showfliers=False, medianprops=dict(color="0.15", lw=1.2),
                        whiskerprops=dict(color="0.35", lw=0.9),
                        capprops=dict(color="0.35", lw=0.9))
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.45)
            patch.set_edgecolor(color)
        for i, (vals, color) in enumerate(zip(data, colors)):
            x = i + rng.uniform(-0.13, 0.13, size=len(vals))
            ax.scatter(x, vals, s=7, color=color, edgecolors="white",
                       linewidths=0.3, zorder=3)
        ax.axhline(chance[p], color="0.45", lw=0.9, ls=":", zorder=0)
        ax.axhline(base[p], color="0.25", lw=1.0, ls="--", zorder=0)
        ax.text(4.45, base[p] + 0.006, "surface", ha="right", va="bottom",
                fontsize=7, color="0.25")
        ax.text(-0.42, chance[p] + 0.006, "chance", ha="left", va="bottom",
                fontsize=7, color="0.45")
        ax.set_title(titles[p])
        ax.set_xticks(range(5))
        ax.set_xticklabels([f"{l}\n{b}" for l, b in zip(labels, best_layers)])
        # shared y-axis across the three panels so heights are comparable;
        # spans conjugation chance (0.33) up to the l_shaped maxima (~0.91)
        ax.set_ylim(0.28, 0.95)
    axes[0].set_ylabel("balanced accuracy")
    fig.tight_layout()
    fig.savefig("paper/figures/boxplots_balanced.pdf", bbox_inches="tight")
    print("  wrote paper/figures/boxplots_balanced.pdf")


def layerwise_boxplot_figure():
    """Per-layer distributions: 3 property panels, 8 layer groups x 5 archs."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs("paper/figures", exist_ok=True)
    df = load_per_run("data/probing/results_stemfinal_lnl_grouped/{arch}/*_results.control.csv")
    df = df[df.probe_type == "linear"]
    ws = pd.read_csv("data/probing/results_lnl_within_stemfinal/summary_10L_90NL.csv")
    bl = pd.concat([pd.read_csv(f) for f in glob.glob(
        "data/probing/results_ngram_baselines_balanced/ngram/*_stemfinal_lnl_results.csv")])
    base = {p: bl[bl.property == p].groupby(
        ["probe_type", "layer_index"]).balanced_accuracy.mean().max() for p in PROPS}
    wsb = ws[(ws.arch == "ngram") & (ws.subset == "all")]
    base["l_shaped"] = max(base["l_shaped"], wsb.balanced_accuracy_mean.max())
    titles = {"l_shaped": "L-shaped (L vs. NL)",
              "conjugation": "conjugation (-ar/-er/-ir)",
              "stem_final_match": "stem-final match"}
    chance = {"l_shaped": 0.5, "conjugation": 1 / 3, "stem_final_match": 0.5}
    plt.rcParams.update({"font.size": 8.5, "axes.titlesize": 9,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.alpha": 0.25, "pdf.fonttype": 42})
    fig, axes = plt.subplots(3, 1, figsize=(9.8, 8.2), sharex=True)
    group_w = len(ARCHS) + 1.5  # slot width per layer group
    for ax, p in zip(axes, ["l_shaped", "conjugation", "stem_final_match"]):
        for li, layer in enumerate(LAY):
            for ai, a in enumerate(ARCHS):
                vals = df[(df.arch == a) & (df.property == p)
                          & (df.layer == layer)].balanced_accuracy.values
                pos = li * group_w + ai
                bp = ax.boxplot([vals], positions=[pos], widths=0.75,
                                patch_artist=True, showfliers=False,
                                medianprops=dict(color="0.15", lw=1.0),
                                whiskerprops=dict(color="0.35", lw=0.8),
                                capprops=dict(color="0.35", lw=0.8))
                bp["boxes"][0].set_facecolor(COLORS[a])
                bp["boxes"][0].set_alpha(0.5)
                bp["boxes"][0].set_edgecolor(COLORS[a])
        ax.axhline(chance[p], color="0.45", lw=0.9, ls=":", zorder=0)
        ax.axhline(base[p], color="0.25", lw=1.0, ls="--", zorder=0)
        ax.text(1.002, base[p], "surface", transform=ax.get_yaxis_transform(),
                fontsize=7, color="0.25", va="center")
        ax.text(1.002, chance[p], "chance", transform=ax.get_yaxis_transform(),
                fontsize=7, color="0.45", va="center")
        ax.set_title(titles[p], loc="left")
        ax.set_ylabel("balanced accuracy")
        for li in range(1, len(LAY)):
            ax.axvline(li * group_w - 1.25, color="0.88", lw=0.6, zorder=0)
    centers = [li * group_w + (len(ARCHS) - 1) / 2 for li in range(len(LAY))]
    axes[-1].set_xticks(centers)
    axes[-1].set_xticklabels(LAY)
    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=COLORS[a], alpha=0.5,
                             edgecolor=COLORS[a]) for a in ARCHS]
    fig.legend(handles, [LABELS[a] for a in ARCHS], loc="upper center",
               ncol=5, frameon=False, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig("paper/figures/boxplots_layerwise.pdf", bbox_inches="tight")
    print("  wrote paper/figures/boxplots_layerwise.pdf")


def table_structure():
    print("\n=== Structure: cell clustering (dec2) + non-ar L/NL (enc3) ===")
    df = load_per_run("data/probing/results_structure/{arch}/*_structure.csv")
    for a in ARCHS:
        d2 = df[(df.arch == a) & (df.layer == "dec2")]
        e3 = df[(df.arch == a) & (df.layer == "enc3")]
        print(f"  {LABELS[a]:<15} P(SBJV|1sgind) L={d2.p_sbjv_1sgind_L.mean():.2f} "
              f"NL={d2.p_sbjv_1sgind_NL.mean():.2f} | "
              f"non-ar L/NL enc3={e3.lshaped_nonar.mean():.2f}"
              f"\u00b1{e3.lshaped_nonar.std():.2f} sel={e3.lshaped_nonar_selectivity.mean():+.2f}")


def table_variance_decomposition():
    print("\n=== Variance decomposition: l_shaped @ dec2, split vs subsample (eta^2) ===")
    df = load_per_run("data/probing/results_stemfinal_lnl_grouped/{arch}/*_results.control.csv")
    df = df[(df.probe_type == "linear") & (df.property == "l_shaped")
            & (df.layer == "dec2")].copy()
    # run id from... not stored; recover from selectivity uniqueness is fragile.
    # Recompute from files instead:
    import glob as _g
    rows = []
    for a in ARCHS:
        for f in _g.glob(f"data/probing/results_stemfinal_lnl_grouped/{a}/*_results.control.csv"):
            run = f.split("10L_90NL_")[1].split("_stemfinal")[0]
            d = pd.read_csv(f)
            d = d[(d.probe_type == "linear") & (d.property == "l_shaped")
                  & (d.layer_type == "decoder") & (d.layer_index == 2)]
            rows.append(dict(arch=a, split=run.split("_")[0], sub=run.split("_")[1],
                             bal=d.balanced_accuracy.iloc[0]))
    dd = pd.DataFrame(rows)

    def eta2(d, factor):
        gm = d.bal.mean()
        ss_t = ((d.bal - gm) ** 2).sum()
        ss_b = sum(len(g) * (g.bal.mean() - gm) ** 2 for _, g in d.groupby(factor))
        return ss_b / ss_t

    for a in ARCHS:
        d = dd[dd.arch == a]
        print(f"  {LABELS[a]:<15} split={eta2(d, 'split'):.2f}  subsample={eta2(d, 'sub'):.2f}")
    pooled = dd.copy()
    pooled["bal"] = pooled.groupby("arch").bal.transform(lambda x: x - x.mean())
    print(f"  pooled          split={eta2(pooled, 'split'):.2f}  subsample={eta2(pooled, 'sub'):.2f}")


if __name__ == "__main__":
    from probing.utils.cli import parse
    args = parse(__doc__)
    table_main()
    table_subset()
    table_positional()
    table_prealt()
    table_transfer()
    table_structure()
    table_variance_decomposition()
    if not args.no_figures:
        boxplot_figure()
        layerwise_boxplot_figure()
