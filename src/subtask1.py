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
    pass
