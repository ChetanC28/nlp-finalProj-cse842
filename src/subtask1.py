# Subtask 1: unsupervised sense induction and diachronic semantic change detection.
# Subtask 1a: Diachronic Sense Assignment.
# Subtask 1b: Sense Dynamics.

import numpy as np
from scipy.spatial.distance import jensenshannon
from scipy.stats import spearmanr
from sklearn.cluster import AgglomerativeClustering

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


def period_distributions(clusters, years, periods):
    """
    Find the sense distribution P_t for each period.
    """
    clusters = np.asarray(clusters)
    years = np.asarray(years)
    unique_clusters = np.unique(clusters)

    distributions = np.zeros((len(periods), len(unique_clusters)))

    for period_idx, period_year in enumerate(periods):
        in_period = (years == period_year)
        period_clusters = clusters[in_period]
        total_in_period = len(period_clusters)

        for cluster_idx, cluster_id in enumerate(unique_clusters):
            count = np.sum(period_clusters == cluster_id)
            distributions[period_idx, cluster_idx] = count / total_in_period

    return distributions


def temporal_change_jsd(distributions):
    """
    Measure semantic change across consecutive periods using Jensen-Shannon Divergence.
    Sum of step-by-step JSDs across adjacent periods, or average.
    """
    total_change = 0.0
    for t in range(len(distributions) - 1):
        total_change += jensenshannon(distributions[t], distributions[t + 1], base=2) ** 2
    return float(total_change)



def evaluate_spearman(predicted_changes, true_changes):
    """
    Compare predicted semantic change rankings with true rankings using Spearman's rho.
    """
    rho, _ = spearmanr(predicted_changes, true_changes)
    return float(rho)


def run_subtask1(df, vectors, split):
    """
    Run Subtask 1a and 1b across words and report macro scores.
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

        pred_dists = period_distributions(clusters, word_df["year"].values, PERIODS)
        true_dists = period_distributions(word_df["sense_id"].values, word_df["year"].values, PERIODS)

        predicted_changes.append(temporal_change_jsd(pred_dists))
        true_changes.append(temporal_change_jsd(true_dists))

    macro_f1 = float(np.mean(bcubed_f1_scores))
    rho = evaluate_spearman(predicted_changes, true_changes)

    print(f"1a {macro_f1:.3f}")
    print(f"1b {rho:.3f}")
    return macro_f1, rho


if __name__ == "__main__":
    import pandas as pd

    df = pd.read_parquet("cache/embeddings_sentences.parquet")
    vectors = np.load("cache/embeddings.npy")
    split = pd.read_csv("splits/word_split.csv")

    run_subtask1(df, vectors, split)

