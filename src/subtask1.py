# Subtask 1: unsupervised sense induction and diachronic semantic change detection.
# Subtask 1a: Diachronic Sense Assignment.
# Subtask 1b: Sense Dynamics.

PERIODS = [1810, 1830, 1850, 1870, 1890, 1910, 1930, 1950, 1970, 1990]


def induce_senses(vectors, threshold):
    """
    Agglomerative clustering with cosine distance 
    (Schlechtweg et al. 2020; Periti & Tahmasebi 2024),
    A single fixed threshold across all words, or a dynamic cutoff picked per word.
    """
    pass


def evaluate_bcubed(gold_senses, pred_clusters):
    """
    Compute BCubed Precision, Recall, and F1 for one word.
    Calculate BCubed F1 per word, then take the macro average 
        across words (following SemEval rules).
    """
    pass


def period_distributions(clusters, years, periods):
    """
    Find the sense distribution P_t for each period.
    """
    pass


def temporal_change_jsd(distributions):
    """
    Measure semantic change across consecutive periods using Jensen-Shannon Divergence.
    Sum of step-by-step JSDs across adjacent periods, or average.
    """
    pass


def evaluate_spearman(pred_changes, gold_changes):
    """
    Compare using Spearman's rho.
    """
    pass


def run_subtask1(df, vectors, split):
    pass
