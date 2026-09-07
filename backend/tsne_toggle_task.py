#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Compute and plot a 2D embedding (t-SNE, UMAP, LDA, PCA) of all toggle_task trials
across all source-combination subfolders. Visualization only (no clustering).

Examples:
    # Plain t-SNE visualization
    python tsne_toggle_task.py --base run/<user>/toggle_task --embed tsne --perplexity 30

    # UMAP with cosine metric (often good for mixed one-hot + numeric)
    python tsne_toggle_task.py --base run/<user>/toggle_task --embed umap --metric cosine \
        --umap-n-neighbors 50 --umap-min-dist 0.2

    # Supervised LDA (maximizes separation by mode)
    python tsne_toggle_task.py --base run/<user>/toggle_task --embed lda

    # Keep only specific feature keys (adjust names to your JSON)
    python tsne_toggle_task.py --base run/<user>/toggle_task --embed tsne \
        --keep-keys active_color,inactive_color,thumb_color,style,thumb_shape,knob_ratio,border_radius
"""

import os
import re
import json
import argparse
import warnings
from glob import glob

import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

# optional install for UMAP
try:
    import umap
except ImportError:
    umap = None

TRIAL_HP_PATTERN = re.compile(r"trial_(\d+)_hyperparams\.json$")
TRIAL_ANY_PATTERN = re.compile(r"trial_(\d+)")


def find_modes(base_dir: str):
    if not os.path.isdir(base_dir):
        raise FileNotFoundError(f"Base directory not found: {base_dir}")
    return [d for d in sorted(os.listdir(base_dir))
            if os.path.isdir(os.path.join(base_dir, d))]


def trials_dir_for_mode(base_dir: str, mode: str) -> str:
    return os.path.join(base_dir, mode, "results", "trials")


def best_dir_for_mode(base_dir: str, mode: str) -> str:
    return os.path.join(base_dir, mode, "results", "best_trial")


def load_trial_params(trials_dir: str):
    out = []
    if not os.path.isdir(trials_dir):
        return out
    for fp in sorted(glob(os.path.join(trials_dir, "trial_*_hyperparams.json"))):
        m = TRIAL_HP_PATTERN.search(os.path.basename(fp))
        if not m:
            continue
        tnum = int(m.group(1))
        try:
            with open(fp, "r") as f:
                params = json.load(f)
            if not isinstance(params, dict):
                params = {"_raw": params}
            out.append((tnum, params, fp))
        except Exception as e:
            warnings.warn(f"Skipping {fp}: {e}")
    return out


def infer_best_trial_number(best_dir: str):
    if not os.path.isdir(best_dir):
        return None
    for fp in sorted(glob(os.path.join(best_dir, "*_result.json"))):
        try:
            with open(fp, "r") as f:
                data = json.load(f)
            trial_id = data.get("trial_id", "")
            m = TRIAL_ANY_PATTERN.search(trial_id) or TRIAL_ANY_PATTERN.search(os.path.basename(fp))
            if m:
                return int(m.group(1))
        except Exception:
            continue
    for fp in sorted(glob(os.path.join(best_dir, "*"))):
        m = TRIAL_ANY_PATTERN.search(os.path.basename(fp))
        if m:
            return int(m.group(1))
    return None


def build_dataset(base_dir: str, keep_keys=None):
    modes = find_modes(base_dir)
    records, meta_mode, meta_tnum, meta_is_best = [], [], [], []
    for mode in modes:
        tdir = trials_dir_for_mode(base_dir, mode)
        bdir = best_dir_for_mode(base_dir, mode)
        best_tnum = infer_best_trial_number(bdir)
        trials = load_trial_params(tdir)
        if not trials:
            continue
        for tnum, params, src in trials:
            rec = dict(params)
            if keep_keys is not None:
                # filter to only requested feature keys
                rec = {k: rec.get(k, None) for k in keep_keys}
            records.append(rec)
            meta_mode.append(mode)
            meta_tnum.append(tnum)
            meta_is_best.append(bool(best_tnum is not None and tnum == best_tnum))
    meta = {
        "mode": np.array(meta_mode),
        "trial_number": np.array(meta_tnum, dtype=int) if meta_tnum else np.array([], dtype=int),
        "is_best": np.array(meta_is_best, dtype=bool) if meta_is_best else np.array([], dtype=bool),
    }
    return records, meta, modes


def encode_features(records):
    """
    DictVectorizer -> dense -> StandardScaler.
    """
    if not records:
        return np.zeros((0, 1)), DictVectorizer(sparse=False), []
    vec = DictVectorizer(sparse=False, sort=True)
    X = vec.fit_transform(records)
    scaler = StandardScaler(with_mean=True, with_std=True)
    Xs = scaler.fit_transform(X)
    return Xs, vec, vec.get_feature_names_out().tolist()


def choose_perplexity(n_points: int, desired: int = 30) -> int:
    if n_points <= 5:
        return max(2, n_points - 1)
    return min(desired, max(5, (n_points - 1) // 3))


def compute_embedding(
    X, meta, method="tsne", metric="euclidean", perplexity=30, random_state=42,
    umap_n_neighbors=30, umap_min_dist=0.1
):
    if X.shape[0] <= 1:
        return np.zeros((X.shape[0], 2))

    if method == "pca":
        return PCA(n_components=2, random_state=random_state).fit_transform(X)

    if method == "lda":
        y = meta["mode"]
        lda = LDA(n_components=2)
        return lda.fit_transform(X, y)

    if method == "umap":
        if umap is None:
            raise ImportError("umap-learn not installed. pip install umap-learn")
        reducer = umap.UMAP(
            n_components=2,
            n_neighbors=umap_n_neighbors,
            min_dist=umap_min_dist,
            metric=metric,
            random_state=random_state
        )
        return reducer.fit_transform(X)

    # t-SNE (default)
    try:
        tsne = TSNE(
            n_components=2,
            perplexity=perplexity,
            init="pca",
            learning_rate="auto",
            max_iter=1000,      # newer sklearn
            random_state=random_state,
            metric=metric,
            verbose=0,
        )
    except TypeError:
        tsne = TSNE(
            n_components=2,
            perplexity=perplexity,
            init="pca",
            learning_rate="auto",
            n_iter=1000,        # older sklearn
            random_state=random_state,
            metric=metric,
            verbose=0,
        )
    return tsne.fit_transform(X)


def plot_embedding(Z, meta, out_path, title="Embedding", method="tsne"):
    if Z.shape[0] == 0:
        print("No trials found. Nothing to plot.")
        return

    plt.figure(figsize=(10, 8), dpi=120)

    # Color by mode (no n=... in labels)
    unique_modes = sorted(set(meta["mode"].tolist()))
    for m in unique_modes:
        mask = (meta["mode"] == m)
        plt.scatter(Z[mask, 0], Z[mask, 1], s=36, alpha=0.8,
                    label=f"{m}", edgecolors="none")

    # Overlay best trials (bigger stars)
    best_mask = meta["is_best"]
    if best_mask.any():
        plt.scatter(Z[best_mask, 0], Z[best_mask, 1], s=400, marker="*",
                    edgecolors="black", linewidths=1.2, alpha=1.0,
                    label="Best trial(s)")

        # Put mode text above each star, larger font, no design number
        y_span = float(Z[:, 1].max() - Z[:, 1].min()) if Z.shape[0] > 1 else 1.0
        text_offset = 0.02 * y_span
        for x, y, m in zip(Z[best_mask, 0], Z[best_mask, 1], meta["mode"][best_mask]):
            plt.text(x, y + text_offset, f"{m}",
                     fontsize=14, fontweight="bold",
                     ha="center", va="bottom")

    plt.title(title)

    # Better axis labels per method
    if method == "pca":
        xl, yl = "PC 1", "PC 2"
    elif method == "lda":
        xl, yl = "LD 1", "LD 2"
    elif method == "umap":
        xl, yl = "UMAP 1", "UMAP 2"
    else:
        xl, yl = "Embedding Dim 1", "Embedding Dim 2"

    plt.xlabel(xl)
    plt.ylabel(yl)

    plt.legend(loc="best", fontsize=9, frameon=True)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"Saved plot to: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True, help="Base path to toggle_task")
    parser.add_argument("--out", default=None, help="Output image path")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed")

    # Embedding choices
    parser.add_argument("--embed", choices=["tsne","umap","lda","pca"], default="tsne",
                        help="2D embedding method for visualization")
    parser.add_argument("--perplexity", type=int, default=30, help="Desired t-SNE perplexity")
    parser.add_argument("--metric", choices=["euclidean","cosine"], default="euclidean",
                        help="Distance metric for t-SNE/UMAP")

    # UMAP knobs
    parser.add_argument("--umap-n-neighbors", type=int, default=30, help="UMAP n_neighbors")
    parser.add_argument("--umap-min-dist", type=float, default=0.1, help="UMAP min_dist")

    # Feature filtering
    parser.add_argument("--keep-keys", type=str, default=None,
                        help="Comma-separated list of feature keys to keep (others dropped).")

    args = parser.parse_args()

    base_dir = os.path.abspath(args.base)
    out_path = args.out or os.path.join(base_dir, f"toggle_task_{args.embed}.svg")

    # Parse keep-keys list
    keep_keys = None
    if args.keep_keys:
        keep_keys = [k.strip() for k in args.keep_keys.split(",") if k.strip()]

    # Build dataset
    records, meta, modes = build_dataset(base_dir, keep_keys=keep_keys)
    Xs, vec, feat_names = encode_features(records)
    n = Xs.shape[0]
    if n == 0:
        print("No trials found across modes. Exiting.")
        return

    # Compute embedding for visualization
    perp = choose_perplexity(n, desired=args.perplexity)
    if args.embed == "tsne" and perp != args.perplexity:
        print(f"Adjusted perplexity from {args.perplexity} to {perp} for N={n}.")
    Z = compute_embedding(
        Xs, meta,
        method=args.embed,
        metric=args.metric,
        perplexity=perp,
        random_state=args.random_state,
        umap_n_neighbors=args.umap_n_neighbors,
        umap_min_dist=args.umap_min_dist
    )

    # Plot and save
    title = f"{args.embed.upper()} embedding (colored by mode)"
    plot_embedding(Z, meta, out_path, title=title, method=args.embed)


if __name__ == "__main__":
    main()
