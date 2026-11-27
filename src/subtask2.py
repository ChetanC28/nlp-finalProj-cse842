# Subtask 2: for one sense of a word, decide which sentences use that sense.
# Our method: the baseline (prototype + one cutoff), plus one cutoff per period that comes from
# a smoothed "sense trajectory". The decisions behind each step are in DECISIONS.md.
#
# Uses (same row order):
#   cache/embeddings.npy               one vector per sentence
#   cache/embeddings_sentences.parquet the cleaned sentences
#
# Run from the repo root: python -m src.subtask2
import json
import os

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from src import baseline
from src.baseline import K, normalize

PERIODS = np.array([1810, 1830, 1850, 1870, 1890, 1910, 1930, 1950, 1970, 1990])
STRENGTHS = [0, 0.5, 1, 2, 3]            # sigma of the smoothing, in periods (0 = no smoothing)
ADJUSTS = [0, 0.05, 0.1, 0.2, 0.3, 0.5]  # how far the cutoff moves (0 = baseline)
MODEL_FILE = "models/temporal.json"        # settings tuned on the pretrained vectors
FT_MODEL_FILE = "models/temporal_ft.json"  # settings tuned on the fine-tuned vectors


# --- Step 1: reference for a sense ---

def build_reference(vectors, sense_rows, k, rng):
    """Average of k random sentences of the sense (any period). Returns the reference and the rows used."""
    examples = rng.choice(sense_rows, k, replace=False)
    return normalize(vectors[examples].mean(axis=0)), examples


# --- Step 2: score sentences ---

def score_sentences(vectors, reference, rows):
    """Cosine similarity of each sentence to the reference (vectors are already normalized)."""
    return vectors[rows] @ reference


def make_cases(df, vectors, words, seed=0):
    """One case per (word, sense), same as baseline.make_cases (same seed, so the same examples),
    plus the period of each scored sentence."""
    rng = np.random.default_rng(seed)
    years = df["year"].values
    cases = []
    for word in words:
        rows = np.where(df["lemma"].values == word)[0]
        senses = df["sense_id"].values[rows]
        for sense in np.unique(senses):
            sense_rows = rows[senses == sense]
            if len(sense_rows) <= K:  # too few sentences to be a target: skip
                continue
            reference, examples = build_reference(vectors, sense_rows, K, rng)
            others = np.setdiff1d(rows, examples)  # the examples are not scored
            cases.append({
                "word": word,
                "scores": score_sentences(vectors, reference, others),
                "labels": senses[np.searchsorted(rows, others)] == sense,
                "periods": years[others],
            })
    return cases


# --- Step 3a: baseline decision ---

def predict_baseline(scores, cutoff):
    """Yes if score >= cutoff. One cutoff for every period."""
    return scores >= cutoff


# --- Step 3b: temporal decision (our method) ---

def sense_trajectory(predictions, periods):
    """Share of the word's sentences in each period that the first pass says YES to.
    NaN for periods where the word has no sentences."""
    trajectory = np.full(len(PERIODS), np.nan)
    for i, period in enumerate(PERIODS):
        in_period = periods == period
        if in_period.any():
            trajectory[i] = predictions[in_period].mean()
    return trajectory


def smooth(trajectory, strength):
    """Gaussian kernel over periods: each period becomes a weighted average of the observed periods,
    with weights falling off with distance in time. strength = sigma in periods; 0 = no smoothing."""
    if strength == 0:
        return trajectory.copy()
    observed = ~np.isnan(trajectory)
    t = np.arange(len(trajectory))
    weights = np.exp(-0.5 * ((t[:, None] - t[None, :]) / strength) ** 2) * observed[None, :]
    smoothed = weights @ np.nan_to_num(trajectory) / weights.sum(axis=1)
    smoothed[~observed] = np.nan
    return smoothed


def period_cutoffs(smoothed, base_cutoff, adjust):
    """One cutoff per period: lower where the sense is more common than its own average, higher where rarer."""
    return base_cutoff - adjust * (smoothed - np.nanmean(smoothed))


def predict_temporal(scores, periods, base_cutoff, strength, adjust):
    """Baseline first pass -> trajectory -> smooth -> per-period cutoffs -> yes/no."""
    first_pass = predict_baseline(scores, base_cutoff)
    smoothed = smooth(sense_trajectory(first_pass, periods), strength)
    cutoffs = period_cutoffs(smoothed, base_cutoff, adjust)
    return scores >= cutoffs[np.searchsorted(PERIODS, periods)]


# --- Step 4: evaluation ---

def evaluate(cases, predict):
    """predict(case) -> yes/no array. Each metric per case, averaged per word, then across words
    (same as baseline.metrics)."""
    rows = []
    for c in cases:
        pred = predict(c)
        rows.append({
            "word": c["word"],
            "f1": f1_score(c["labels"], pred, zero_division=0),
            "precision": precision_score(c["labels"], pred, zero_division=0),
            "recall": recall_score(c["labels"], pred, zero_division=0),
            "accuracy": accuracy_score(c["labels"], pred),
        })
    return pd.DataFrame(rows).groupby("word").mean().mean().round(3).to_dict()


# --- Step 5: tuning and running ---

def tune(train_cases, base_cutoff):
    """Base cutoff stays the baseline's; try every (strength, adjust) and keep the best train F1.
    On a tie the first one wins, i.e. the least smoothing / smallest adjustment."""
    best, best_f1 = (0, 0), -1
    for strength in STRENGTHS:
        for adjust in ADJUSTS:
            f1 = evaluate(train_cases, lambda c: predict_temporal(c["scores"], c["periods"], base_cutoff, strength, adjust))["f1"]
            print(f"  strength={strength} adjust={adjust}: train F1 {f1}")
            if f1 > best_f1:
                best, best_f1 = (strength, adjust), f1
    return best


def run_subtask2(df, vectors, split, final_test=False, finetuned=False):
    """Tune, then compare baseline vs temporal on dev (and on test only if final_test=True).
    finetuned=True: the vectors come from the fine-tuned model, so tuning uses the holdout words only."""
    # base cutoff = the baseline's learned cutoff for these vectors, trained if missing
    if os.path.exists(baseline.model_file(finetuned)):
        base_cutoff = json.load(open(baseline.model_file(finetuned)))["cutoff"]
    else:
        base_cutoff = baseline.train(df, vectors, split, finetuned)
    vectors = normalize(vectors.astype(np.float32))

    strength, adjust = tune(make_cases(df, vectors, baseline.tuning_words(split, finetuned)), base_cutoff)
    print(f"base cutoff {base_cutoff} | best strength {strength} | best adjust {adjust}")
    os.makedirs("models", exist_ok=True)
    json.dump({"base_cutoff": float(base_cutoff), "strength": strength, "adjust": adjust, "k": K},
              open(FT_MODEL_FILE if finetuned else MODEL_FILE, "w"))

    names = ["dev", "test"] if final_test else ["dev"]
    results = {}
    for name in names:
        cases = make_cases(df, vectors, split.loc[split["split"] == name, "lemma"])
        results[name] = {
            "baseline": evaluate(cases, lambda c: predict_baseline(c["scores"], base_cutoff)),
            "temporal": evaluate(cases, lambda c: predict_temporal(c["scores"], c["periods"], base_cutoff, strength, adjust)),
        }
        print(f"{name} baseline: {results[name]['baseline']}")
        print(f"{name} temporal: {results[name]['temporal']}")
    return results


if __name__ == "__main__":
    # python -m src.subtask2              -> pretrained vectors
    # python -m src.subtask2 --finetuned  -> fine-tuned vectors (run python -m src.embed first)
    import sys
    finetuned = "--finetuned" in sys.argv
    name = "embeddings_ft" if finetuned else "embeddings"
    df = pd.read_parquet(f"cache/{name}_sentences.parquet")
    vectors = np.load(f"cache/{name}.npy")
    split = pd.read_csv("splits/word_split.csv")
    run_subtask2(df, vectors, split, finetuned=finetuned)
