"""Subtask 1 clustering and temporal experiments on cached XL-LEXEME embeddings."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from scipy.stats import spearmanr
from sklearn.cluster import AgglomerativeClustering

PERIODS = np.array([1810, 1830, 1850, 1870, 1890, 1910, 1930, 1950, 1970, 1990])


def bcubed(gold, predicted):
    """Compute per-word BCubed precision, recall, and F1 from a contingency table."""
    gold_codes, gold_keys = pd.factorize(np.asarray(gold), sort=False)
    pred_codes, pred_keys = pd.factorize(np.asarray(predicted), sort=False)
    n = len(gold_codes)
    if n == 0:
        raise ValueError("Cannot score an empty word")
    counts = np.zeros((len(pred_keys), len(gold_keys)), dtype=np.int64)
    np.add.at(counts, (pred_codes, gold_codes), 1)
    numer = counts.astype(np.float64) ** 2
    p = float((numer / counts.sum(axis=1)[:, None]).sum() / n)
    r = float((numer / counts.sum(axis=0)[None, :]).sum() / n)
    f = 2 * p * r / (p + r) if p + r > 0 else 0.0
    return p, r, f


def normalized(x):
    x = np.asarray(x, dtype=np.float32)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    if not np.isfinite(norms).all() or (norms == 0).any():
        raise ValueError("Found NaN/Inf or zero XL-LEXEME embedding; inspect data.")
    return x / norms


def cluster(x, threshold):
    if len(x) <= 1:
        return np.zeros(len(x), dtype=np.int32)
    return AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=float(threshold),
        metric="cosine",
        linkage="average",
    ).fit_predict(x)


def sense_distributions(labels, years, periods=PERIODS):
    """Return T x K frequencies and T-element observed mask.

    For missing periods, rows are NaN (not zero-distribution, and never
    automatically interpreted as semantic change).
    """
    ids, unique = pd.factorize(np.asarray(labels), sort=False)
    counts = np.zeros((len(periods), len(unique)), dtype=np.float64)
    for t, period in enumerate(periods):
        selected = years == period
        if selected.any():
            counts[t] = np.bincount(ids[selected], minlength=len(unique))
    sizes = counts.sum(axis=1)
    observed = sizes > 0
    dist = np.full_like(counts, np.nan, dtype=np.float64)
    dist[observed] = counts[observed] / sizes[observed, None]
    return dist, observed


def jsd_change(labels, years, mode="observed"):
    """Sum Jensen-Shannon DIVERGENCES (SciPy returns sqrt divergence).

    observed: original repo's skip-missing behavior, even across large gaps.
    adjacent: compare only genuinely neighboring calendar periods (diagnostic).
    Official SemEval missing-period policy should be checked separately.
    """
    dist, observed = sense_distributions(labels, years)
    if mode == "observed":
        idx = np.flatnonzero(observed)
        pairs = zip(idx[:-1], idx[1:])
    elif mode == "adjacent":
        pairs = ((i, i + 1) for i in range(len(PERIODS) - 1)
                 if observed[i] and observed[i + 1])
    else:
        raise ValueError(f"unknown mode {mode}")
    total = 0.0
    for a, b in pairs:
        total += float(jensenshannon(dist[a], dist[b], base=2) ** 2)
    return total


def smooth_reassign(x, original, years, strength, window=0.5, margin=0.06):
    """Time-aware, no-gold-label reassignment of ambiguous usages.

    - Initial groups and centroids come from unsupervised AHC only.
    - Temporal priors come from initial PREDICTED groups in nearby periods.
    - An observation's own count is subtracted from its temporal prior.
    - A candidate cannot be too much less semantically similar to its centroid.
    - strength == 0 is exactly identical to the original baseline.

    Strength scales log(local sense share / word-wide sense share). """
    labels = np.asarray(original, dtype=np.int32)
    if strength == 0 or len(labels) <= 1:
        return labels.copy()
    unique, cluster_ids = np.unique(labels, return_inverse=True)
    k = len(unique)
    if k <= 1:
        return labels.copy()
    d = x.shape[1]
    centroids = np.empty((k, d), dtype=np.float32)
    for i in range(k):
        centroids[i] = x[cluster_ids == i].mean(axis=0)
    norm = np.linalg.norm(centroids, axis=1, keepdims=True)
    centroids = centroids / np.maximum(norm, 1e-12)
    similarity = x @ centroids.T

    # Estimate temporal priors from predicted clusters on the fixed time grid.
    years_idx = np.searchsorted(PERIODS, years)
    counts = np.zeros((len(PERIODS), k), dtype=np.float64)
    np.add.at(counts, (years_idx, cluster_ids), 1)
    kernel = np.eye(len(PERIODS), dtype=np.float64)
    for t in range(len(PERIODS) - 1):
        kernel[t, t + 1] = window
        kernel[t + 1, t] = window
    aggregate = kernel @ counts
    totals = aggregate.sum(axis=1)
    global_share = counts.sum(axis=0) / len(labels)

    # Exclude the current occurrence from its estimated local prior.
    own_weight = np.diag(kernel)[years_idx]
    local_counts = aggregate[years_idx].copy()
    local_counts[np.arange(len(labels)), cluster_ids] -= own_weight
    local_total = totals[years_idx] - own_weight
    pseudo = 2.0  # global-prior pseudo-observations stabilize tiny periods
    prior = (local_counts + pseudo * global_share[None, :]) / (local_total[:, None] + pseudo)
    log_ratio = np.log(np.maximum(prior, 1e-12) / np.maximum(global_share[None, :], 1e-12))
    adjusted = similarity + float(strength) * log_ratio
    candidate = adjusted.argmax(axis=1)
    own = np.arange(len(labels))
    accept = (candidate != cluster_ids) & (
        similarity[own, candidate] >= similarity[own, cluster_ids] - margin
    ) & (
        adjusted[own, candidate] > adjusted[own, cluster_ids]
    )
    out = labels.copy()
    out[accept] = unique[candidate[accept]]
    return out


def score_rho(pred, gold):
    if len(pred) < 2 or len(np.unique(pred)) < 2 or len(np.unique(gold)) < 2:
        return 0.0
    rho = spearmanr(pred, gold).statistic
    return float(rho) if np.isfinite(rho) else 0.0


def evaluate(df, vector_file, words, configurations):
    """Evaluate multiple (threshold, temporal strength) pairs efficiently.

    Each word is clustered once per requested threshold, independent of the
    number of temporal strengths. Records permit later error analysis.
    """
    words = [str(w) for w in words]
    want = set(words)
    configurations = list(dict.fromkeys((float(t), float(s)) for t, s in configurations))
    by_thr = {}
    for t, strength in configurations:
        by_thr.setdefault(t, []).append(strength)

    vectors = np.load(vector_file, mmap_mode="r")
    if vectors.ndim != 2 or vectors.shape[0] != len(df):
        raise ValueError(f"Embedding rows {vectors.shape} do not match metadata rows {len(df)}")
    positions = df.groupby("lemma", sort=False).indices
    rows = []
    for j, word in enumerate(words, start=1):
        if word not in positions:
            raise ValueError(f"Split word {word!r} not present in embeddings table")
        loc = positions[word]
        x = normalized(vectors[loc])
        group = df.iloc[loc]
        gold = group["sense_id"].to_numpy()
        years = group["year"].to_numpy(dtype=np.int64)
        gold_observed = jsd_change(gold, years, "observed")
        gold_adjacent = jsd_change(gold, years, "adjacent")
        missing = int(len(PERIODS) - len(np.unique(years)))
        for threshold, strengths in by_thr.items():
            initial = cluster(x, threshold)
            for strength in strengths:
                predicted = smooth_reassign(x, initial, years, strength)
                precision, recall, f1 = bcubed(gold, predicted)
                rows.append({
                    "word": word, "n_sentences": len(loc),
                    "missing_periods": missing, "threshold": threshold,
                    "strength": strength, "bcubed_precision": precision,
                    "bcubed_recall": recall, "bcubed_f1": f1,
                    "gold_change_observed": gold_observed,
                    "pred_change_observed": jsd_change(predicted, years, "observed"),
                    "gold_change_adjacent": gold_adjacent,
                    "pred_change_adjacent": jsd_change(predicted, years, "adjacent"),
                    "n_clusters": len(np.unique(predicted)),
                    "n_changed": int((predicted != initial).sum()),
                })
        if j % 50 == 0 or j == len(words):
            print(f"  processed {j}/{len(words)} words", flush=True)
    return pd.DataFrame.from_records(rows)


def summarize(rows):
    result = []
    for (threshold, strength), group in rows.groupby(["threshold", "strength"], sort=True):
        result.append({
            "threshold": threshold, "strength": strength,
            "words": len(group),
            "bcubed_f1": group["bcubed_f1"].mean(),
            "bcubed_precision": group["bcubed_precision"].mean(),
            "bcubed_recall": group["bcubed_recall"].mean(),
            "rho_observed": score_rho(group["pred_change_observed"], group["gold_change_observed"]),
            "rho_adjacent": score_rho(group["pred_change_adjacent"], group["gold_change_adjacent"]),
            "words_missing_periods": int((group["missing_periods"] > 0).sum()),
            "changed_sentences": int(group["n_changed"].sum()),
            "sentences": int(group["n_sentences"].sum()),
        })
    return pd.DataFrame(result)


def save_plots(train_static, train_temporal, baseline_words, out):
    """Create small publication-friendly diagnostics from the real run."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is unavailable; skipping optional plots")
        return

    fig, ax = plt.subplots(figsize=(6, 4))
    z = train_static.sort_values("threshold")
    ax.plot(z["threshold"], z["bcubed_f1"], "o-")
    ax.set(xlabel="AHC cosine distance threshold", ylabel="Macro BCubed F1",
           title="Training-word clustering threshold comparison")
    fig.tight_layout()
    fig.savefig(out / "train_threshold_f1.png", dpi=180)
    plt.close(fig)

    z = train_temporal.sort_values("strength")
    for metric, suffix, ylabel in [("bcubed_f1", "f1", "Macro BCubed F1"),
                                   ("rho_observed", "rho", "Spearman rho")]:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(z["strength"], z[metric], "o-")
        ax.set(xlabel="Temporal prior strength", ylabel=ylabel,
               title=f"Training-word temporal adjustment ({suffix})")
        fig.tight_layout()
        fig.savefig(out / f"train_temporal_{suffix}.png", dpi=180)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(baseline_words["bcubed_f1"], bins=18)
    ax.axvline(baseline_words["bcubed_f1"].mean(), linestyle="--", label="Mean")
    ax.set(xlabel="BCubed F1", ylabel="Number of development words",
           title="Subtask 1a baseline word-level scores")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "dev_baseline_bcubed_histogram.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(baseline_words["gold_change_observed"],
               baseline_words["pred_change_observed"], alpha=0.7)
    ax.set(xlabel="Gold-label sample JSD sum", ylabel="Predicted JSD sum",
           title="Subtask 1b baseline change-score agreement")
    fig.tight_layout()
    fig.savefig(out / "dev_baseline_change_scatter.png", dpi=180)
    plt.close(fig)


def verify_data(df, vector_file, split):
    required = {"lemma", "sense_id", "year", "text", "start", "end"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Embeddings parquet missing required columns: {sorted(missing)}")
    if not {"lemma", "split"}.issubset(split.columns):
        raise ValueError("Split CSV must have lemma,split columns")
    sizes = np.load(vector_file, mmap_mode="r")
    if len(df) != sizes.shape[0]:
        raise ValueError("Row alignment failure: embeddings count != metadata row count")
    if len(df) == 0 or sizes.ndim != 2:
        raise ValueError("Empty data or non-matrix embeddings")
    df["year"] = pd.to_numeric(df["year"], errors="raise").astype(int)
    unknown = np.setdiff1d(df["year"].unique(), PERIODS)
    if len(unknown):
        raise ValueError(f"Unexpected years, expected period start years only: {unknown}")
    if df[["lemma", "sense_id", "year"]].isna().any().any():
        raise ValueError("Missing lemma, sense_id, or year")
    cleaned_words = set(df.lemma.astype(str))
    split_words = list(split.lemma.astype(str))
    if len(split_words) != len(set(split_words)):
        raise ValueError("Duplicate word(s) in split file")
    missing_split = set(split_words) - cleaned_words
    if missing_split:
        raise ValueError(f"Split still contains absent words after reconciliation: {sorted(missing_split)[:10]}")
    unassigned = cleaned_words - set(split_words)
    if unassigned:
        print(f"NOTE: {len(unassigned)} embedding word(s) have no split assignment and are excluded from evaluation: {sorted(unassigned)[:10]}")
    print(f"Data OK: {len(df):,} rows x {sizes.shape[1]} dims, {len(cleaned_words)} words")
    print("Split counts:", split["split"].value_counts().to_dict())
    return df


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False))


def reconcile_split(df, split, output_dir):
    """Keep the committed split assignments for words that have cached vectors.

    Do NOT regenerate a split or reassign previously unseen words. Report all
    mismatches so that baselines remain auditable. Embeddings are untouched.
    """
    present = set(df["lemma"].astype(str))
    split_set = set(split["lemma"].astype(str))
    absent = sorted(split_set - present)
    unassigned = sorted(present - split_set)
    excluded_by_split = split.loc[split["lemma"].astype(str).isin(absent), ["lemma", "split"]]
    valid = split.loc[split["lemma"].astype(str).isin(present)].copy()
    payload = {
        "words_in_embeddings": len(present),
        "words_in_committed_split": len(split_set),
        "words_in_effective_split": int(len(valid)),
        "missing_from_embeddings": excluded_by_split.to_dict(orient="records"),
        "without_split_assignment": unassigned,
        "effective_split_counts": valid["split"].value_counts().to_dict(),
        "policy": "Keep existing assignments; exclude unavailable or unassigned words; do not regenerate splits or touch embeddings",
    }
    write_json(output_dir / "split_reconciliation.json", payload)
    print("Split reconciliation:", payload, flush=True)
    if valid.empty or not (valid["split"] == "train").any() or not (valid["split"] == "dev").any():
        raise ValueError("Effective split has no training or development words")
    return valid, payload


def pick_best(df, metric):
    # Select hyperparameters using training words.
    ordered = df.sort_values([metric, "strength"], ascending=[False, True])
    return ordered.iloc[0]


def run(args):
    data_dir = Path(args.data_dir).expanduser().resolve()
    root = Path(args.repo_root).expanduser().resolve()
    out = Path(args.output_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    parq = data_dir / "embeddings_sentences.parquet"
    emb = data_dir / "embeddings.npy"
    if not parq.exists() or not emb.exists():
        raise FileNotFoundError(f"Expected {parq} and {emb}")
    df = pd.read_parquet(parq)
    split = pd.read_csv(root / "splits/word_split.csv")
    split, split_info = reconcile_split(df, split, out)
    verify_data(df, emb, split)
    dev_words = split.loc[split.split == "dev", "lemma"].astype(str).tolist()
    test_words = split.loc[split.split == "test", "lemma"].astype(str).tolist()
    train_words = split.loc[split.split == "train", "lemma"].astype(str).tolist()
    # Deterministic tune subset, without looking at the held-out dev/test words.
    chosen = np.random.default_rng(args.seed).choice(
        np.array(train_words), size=min(args.train_words, len(train_words)), replace=False
    ).tolist()

    print("\n1) Original baseline on ALL dev words, threshold=0.35")
    original_dev = evaluate(df, emb, dev_words, [(0.35, 0.0)])
    baseline = summarize(original_dev)
    original_dev.to_csv(out / "baseline_dev_per_word.csv", index=False)
    baseline.to_csv(out / "baseline_dev_summary.csv", index=False)
    print(baseline.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print("\n2) Threshold sweep on fixed TRAIN words only")
    thresholds = [float(t) for t in args.thresholds]
    strengths = [float(s) for s in args.strengths]
    if not (0.0 in strengths and 0.35 in thresholds):
        raise ValueError("Thresholds must contain 0.35 and strengths must contain 0.0")
    train_static = evaluate(df, emb, chosen, [(t, 0.0) for t in thresholds])
    train_static_sum = summarize(train_static)
    train_static_sum.to_csv(out / "train_threshold_sweep.csv", index=False)
    best_t = float(train_static_sum.sort_values(["bcubed_f1", "threshold"], ascending=[False, True]).iloc[0].threshold)
    print(train_static_sum.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("Train-selected static threshold:", best_t)

    print("\n3) Temporal reassignment strengths on TRAIN only")
    train_temporal = evaluate(df, emb, chosen, [(best_t, s) for s in strengths])
    train_temporal_sum = summarize(train_temporal)
    train_temporal_sum.to_csv(out / "train_temporal_sweep.csv", index=False)
    best_f1 = float(pick_best(train_temporal_sum, "bcubed_f1").strength)
    best_rho = float(pick_best(train_temporal_sum, "rho_observed").strength)
    print(train_temporal_sum.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("Train-selected strengths for 1a F1 / 1b rho:", best_f1, best_rho)

    configs = list(dict.fromkeys([(0.35, 0.0), (best_t, 0.0),
                                  (best_t, best_f1), (best_t, best_rho)]))
    print("\n4) Final DEV comparison (all dev words)")
    dev = evaluate(df, emb, dev_words, configs)
    dev.to_csv(out / "dev_per_word.csv", index=False)
    dev_sum = summarize(dev)
    dev_sum.to_csv(out / "dev_comparison.csv", index=False)
    print(dev_sum.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    save_plots(train_static_sum, train_temporal_sum, original_dev, out)

    chosen_config = {
        "baseline_threshold": 0.35, "train_tuned_threshold": best_t,
        "strength_for_subtask_1a": best_f1,
        "strength_for_subtask_1b": best_rho,
    }
    write_json(out / "selected_train_parameters.json", chosen_config)
    if args.final_test:
        print("\n5) ONE held-out TEST evaluation, with frozen train-selected settings")
        test = evaluate(df, emb, test_words, configs)
        test.to_csv(out / "test_per_word.csv", index=False)
        test_sum = summarize(test)
        test_sum.to_csv(out / "test_comparison.csv", index=False)
        print(test_sum.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    else:
        print("\n5) Skipping test. Add --final-test only after choosing experiment settings.")

    meta = {
        "utc_created": datetime.now(timezone.utc).isoformat(),
        "source": "synthetic generated sentence sense labels (not the official SemEval scorer)",
        "data_rows": int(len(df)), "embedding_dims": int(np.load(emb, mmap_mode="r").shape[1]),
        "n_words": int(df.lemma.nunique()), "train_tuning_words": len(chosen),
        "seed": args.seed, "thresholds": thresholds, "strengths": strengths,
        "split_reconciliation": split_info,
        "final_test_evaluated": bool(args.final_test),
        "missing_period_policies": ["observed(original)", "adjacent(diagnostic)"],
        "parameters": chosen_config,
        "notes": [
            "Original baseline reproduced at strength 0, threshold 0.35.",
            "Temporal priors use predicted labels only, with own-example exclusion.",
            "Both subtask metrics are evaluated; 1a and 1b may prefer different strengths.",
            "Gold for 1b uses generated sentence annotations; metadata curves are not used.",
            "Dataset frequencies vs metadata are NOT assumed to match or mismatch.",
            "Do not treat diagnostic adjacent-mode as official scoring protocol.",
        ],
    }
    write_json(out / "run_manifest.json", meta)
    print("\nResults saved to:", out)
    return meta


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", required=True, help="Directory with embeddings.npy and embeddings_sentences.parquet")
    p.add_argument("--repo-root", default=".")
    p.add_argument("--output-dir", default="results/subtask1")
    p.add_argument("--train-words", type=int, default=180, help="Fixed subset of training words for tuning")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--thresholds", type=float, nargs="+", default=[0.25, 0.30, 0.35, 0.40, 0.45])
    p.add_argument("--strengths", type=float, nargs="+", default=[0.0, 0.02, 0.05, 0.10, 0.20, 0.35])
    p.add_argument("--final-test", action="store_true", help="Evaluate test once, without tuning on it")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
