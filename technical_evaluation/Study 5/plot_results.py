# plot_results.py
import argparse
import pickle
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def load_distances(pkl_path):
    with open(pkl_path, "rb") as f:
        res = pickle.load(f)
    # distances is a dict like {"M=1": [...], "M=2": [...], ...}
    return res.get("distances", {}), res

def parse_M_keys(d):
    # returns sorted list of ints for M values present
    Ms = []
    for k in d.keys():
        try:
            Ms.append(int(k.split("=")[1]))
        except Exception:
            pass
    return sorted(Ms)

def main():
    ap = argparse.ArgumentParser(description="Grouped boxplots: single vs multi surrogate.")
    ap.add_argument("--single", required=True, help="Path to single-surrogate results .pkl")
    ap.add_argument("--multi",  required=True, help="Path to multi-surrogate results .pkl")
    ap.add_argument("--out", default="boxplot_comparison", help="Output filename (no extension or .pdf/.png)")
    ap.add_argument("--title", default="Comparison of Single- and Multi-Surrogate Optimization")
    ap.add_argument("--ylim", type=float, nargs=2, default=[0.0, 1.0], help="y-axis limits, e.g., 0 1")
    ap.add_argument("--figsize", type=float, nargs=2, default=[10, 6], help="Figure size, e.g., 10 6")
    ap.add_argument("--showmeans", action="store_true", help="Show mean markers on boxplots")
    args = ap.parse_args()

    # Fonts
    plt.rcParams.update({
        "font.size": 14,
        "axes.labelsize": 16,
        "axes.titlesize": 16,
        "legend.fontsize": 14,
        "xtick.labelsize": 13,
        "ytick.labelsize": 13,
    })

    dist_single, meta_single = load_distances(args.single)
    dist_multi,  meta_multi  = load_distances(args.multi)

    Ms_single = parse_M_keys(dist_single)
    Ms_multi  = parse_M_keys(dist_multi)
    Ms_both   = sorted(set(Ms_single).intersection(Ms_multi))

    if not Ms_both:
        raise ValueError("No overlapping M values between the two PKLs.")

    dropped_single = sorted(set(Ms_single) - set(Ms_both))
    dropped_multi  = sorted(set(Ms_multi)  - set(Ms_both))
    if dropped_single:
        print(f"[info] Single-only M values ignored: {dropped_single}")
    if dropped_multi:
        print(f"[info] Multi-only M values ignored:  {dropped_multi}")

    # Prepare data in matching order
    labels = [str(M) for M in Ms_both]
    data_single = [dist_single[f"M={M}"] for M in Ms_both]
    data_multi  = [dist_multi[f"M={M}"]  for M in Ms_both]

    # Plot
    fig, ax = plt.subplots(figsize=tuple(args.figsize))

    # even spacing per M, two boxes per group
    idx = np.arange(len(labels))
    offset = 0.35
    width  = 0.6

    pos_single = idx * 2.0 - offset
    pos_multi  = idx * 2.0 + offset

    bp1 = ax.boxplot(
        data_single,
        positions=pos_single,
        widths=width,
        patch_artist=True,
        showmeans=args.showmeans,
        manage_ticks=False,
    )
    bp2 = ax.boxplot(
        data_multi,
        positions=pos_multi,
        widths=width,
        patch_artist=True,
        showmeans=args.showmeans,
        manage_ticks=False,
    )

    # Simple coloring to distinguish sets
    for patch in bp1["boxes"]:
        patch.set_facecolor("#1f77b4")  # single
        patch.set_alpha(0.6)
    for patch in bp2["boxes"]:
        patch.set_facecolor("#ff7f0e")  # multi
        patch.set_alpha(0.6)

    # Style medians to be clear
    for med in bp1["medians"]:
        med.set_linewidth(2.0)
    for med in bp2["medians"]:
        med.set_linewidth(2.0)

    # X axis grouped at the center between the two boxes
    ax.set_xticks(idx * 2.0)
    ax.set_xticklabels(labels)
    ax.set_xlabel("Number of Sources (M)")
    ax.set_ylabel("Euclidean Distance to Global Min")
    ax.set_title(args.title)
    ax.set_ylim(args.ylim[0], args.ylim[1])

    # Legend
    ax.legend([bp1["boxes"][0], bp2["boxes"][0]], ["Single-Surrogate", "Multi-Surrogate"], loc="upper right")

    plt.tight_layout()

    # Output filenames
    out = Path(args.out)
    if out.suffix.lower() in {".pdf", ".png"}:
        pdf_path = out
        png_path = out.with_suffix(".png") if out.suffix.lower() != ".png" else out
    else:
        pdf_path = out.with_suffix(".pdf")
        png_path = out.with_suffix(".png")

    plt.savefig(pdf_path, dpi=300)
    plt.savefig(png_path, dpi=300)
    print(f"Saved: {pdf_path} and {png_path}")
    plt.show()


if __name__ == "__main__":
    main()