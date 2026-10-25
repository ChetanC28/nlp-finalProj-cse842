# Subtask 1: unsupervised sense induction and diachronic semantic change detection.
# Subtask 1a: Diachronic Sense Assignment.
# Subtask 1b: Sense Dynamics.

import numpy as np
from scipy.spatial.distance import jensenshannon
from scipy.stats import spearmanr
from sklearn.cluster import AgglomerativeClustering
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PERIODS = [1810, 1830, 1850, 1870, 1890, 1910, 1930, 1950, 1970, 1990]


def induce_senses(vectors, threshold=0.35):
    """
    Agglomerative clustering with cosine distance 
    (Schlechtweg et al. 2020; Periti & Tahmasebi 2024),
    A single fixed threshold across all words, or a dynamic cutoff picked per word.
    """
    if len(vectors) == 0:
        return np.array([])
    if len(vectors) == 1:
        return np.array([0])

    clusterer = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=threshold,
        metric="cosine",
        linkage="average",
    )
    return clusterer.fit_predict(vectors)


def evaluate_bcubed(true_senses, predicted_clusters):
    """
    Compute BCubed Precision, Recall, and F1 for one word.
    Calculate BCubed F1 per word, then take the macro average 
        across words (following SemEval rules).
    """
    true_senses = np.asarray(true_senses)
    predicted_clusters = np.asarray(predicted_clusters)

    same_predicted = (predicted_clusters[:, None] == predicted_clusters[None, :])
    same_true = (true_senses[:, None] == true_senses[None, :])
    correct_matches = same_predicted & same_true

    item_precisions = correct_matches.sum(axis=1) / same_predicted.sum(axis=1)
    item_recalls = correct_matches.sum(axis=1) / same_true.sum(axis=1)

    precision = float(np.mean(item_precisions))
    recall = float(np.mean(item_recalls))
    f1_score = 2.0 * (precision * recall) / (precision + recall)

    return precision, recall, f1_score


def period_distributions(clusters, years, periods=None):
    """
    Find the sense distribution P_t for each period.
    Only includes periods that contain usages to avoid zero division.
    """
    clusters = np.asarray(clusters)
    years = np.asarray(years)
    unique_clusters = np.unique(clusters)

    if periods is None:
        periods = np.sort(np.unique(years))

    distributions = []
    for period_year in periods:
        in_period = (years == period_year)
        period_clusters = clusters[in_period]
        total_in_period = len(period_clusters)

        if total_in_period > 0:
            dist = np.zeros(len(unique_clusters))
            for cluster_idx, cluster_id in enumerate(unique_clusters):
                dist[cluster_idx] = np.sum(period_clusters == cluster_id) / total_in_period
            distributions.append(dist)

    if len(distributions) == 0:
        return np.zeros((0, len(unique_clusters)))

    return np.array(distributions)


def temporal_change_jsd(distributions):
    """
    Measure semantic change across consecutive periods using Jensen-Shannon Divergence.
    Sum of step-by-step JSDs across adjacent periods, or average.
    """
    distributions = np.asarray(distributions)
    if len(distributions) <= 1:
        return 0.0

    total_change = 0.0
    for t in range(len(distributions) - 1):
        jsd = jensenshannon(distributions[t], distributions[t + 1], base=2)
        if not np.isnan(jsd):
            total_change += float(jsd ** 2)

    return float(total_change)


def evaluate_spearman(predicted_changes, true_changes):
    """
    Compare predicted semantic change rankings with true rankings using Spearman's rho.
    """
    predicted_changes = np.asarray(predicted_changes)
    true_changes = np.asarray(true_changes)

    valid = ~(np.isnan(predicted_changes) | np.isnan(true_changes))
    if np.sum(valid) < 2:
        return 0.0

    rho, _ = spearmanr(predicted_changes[valid], true_changes[valid])
    return float(0.0 if np.isnan(rho) else rho)


def plot_subtask1_results(bcubed_f1_scores, true_changes, predicted_changes, output_dir="plots"):
    """
    Generate and save evaluation plots for Subtask 1a and Subtask 1b.
    """
    os.makedirs(output_dir, exist_ok=True)

    true_changes = np.asarray(true_changes)
    predicted_changes = np.asarray(predicted_changes)
    bcubed_f1_scores = np.asarray(bcubed_f1_scores)

    # 1. Subtask 1a
    if len(bcubed_f1_scores) > 0:
        macro_f1 = float(np.mean(bcubed_f1_scores))
        plt.figure()
        plt.hist(bcubed_f1_scores)
        plt.axvline(macro_f1, linestyle="--", label=f"macro_f1 ({macro_f1:.3f})")
        plt.xlabel("bcubed_f1_scores")
        plt.ylabel("count")
        plt.title("Subtask 1a: bcubed_f1_scores")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "subtask1a_bcubed_distribution.png"))
        plt.close()

    # 2. Subtask 1b
    valid = ~(np.isnan(true_changes) | np.isnan(predicted_changes))
    if np.sum(valid) >= 2:
        rho, _ = spearmanr(predicted_changes[valid], true_changes[valid])
        plt.figure()
        plt.scatter(true_changes[valid], predicted_changes[valid])
        plt.xlabel("true_changes")
        plt.ylabel("predicted_changes")
        plt.title(f"Subtask 1b: predicted_changes vs true_changes (rho = {rho:.3f})")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "subtask1b_change_correlation.png"))
        plt.close()


def run_subtask1(df, vectors, split, save_plots=True):
    """
    Run Subtask 1a and 1b across words, report macro scores, and optionally save plots.
    """
    dev_words = split.loc[split["split"] == "dev", "lemma"].values
    words = [w for w in df["lemma"].unique() if w in dev_words]
    if len(words) == 0:
        words = df["lemma"].unique()

    bcubed_f1_scores = []
    predicted_changes = []
    true_changes = []

    for word in words:
        mask = (df["lemma"] == word).to_numpy()
        word_df = df[mask].reset_index(drop=True)
        word_vectors = vectors[mask]

        clusters = induce_senses(word_vectors)
        _, _, f1 = evaluate_bcubed(word_df["sense_id"].values, clusters)
        bcubed_f1_scores.append(f1)

        obs_years = np.sort(word_df["year"].unique())
        pred_dists = period_distributions(clusters, word_df["year"].values, obs_years)
        true_dists = period_distributions(word_df["sense_id"].values, word_df["year"].values, obs_years)

        predicted_changes.append(temporal_change_jsd(pred_dists))
        true_changes.append(temporal_change_jsd(true_dists))

    macro_f1 = float(np.mean(bcubed_f1_scores))
    rho = evaluate_spearman(predicted_changes, true_changes)

    print(f"1a {macro_f1}")
    print(f"1b {rho}")

    if save_plots:
        plot_subtask1_results(
            bcubed_f1_scores,
            true_changes,
            predicted_changes,
        )
        print("Saved plots to plots/")

    return macro_f1, rho


if __name__ == "__main__":
    import pandas as pd

    df = pd.read_parquet("cache/embeddings_sentences.parquet")
    vectors = np.load("cache/embeddings.npy")
    split = pd.read_csv("splits/word_split.csv")

    run_subtask1(df, vectors, split)

