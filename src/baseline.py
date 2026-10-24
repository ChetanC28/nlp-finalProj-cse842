# Most basic Subtask 2 model: prototype + one fixed cutoff. No year information.
#
# train():   find the cutoff that gives the best F1 on the train words, save it.
# predict(): for a new input (example sentences of a sense + new sentences), say yes/no.
import json
import os

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

K = 3  # PLACEHOLDER: number of example sentences per sense (decision 1, not decided yet)
MODEL_FILE = "models/baseline.json"


def normalize(v):
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def make_cases(df, vectors, words, seed=0):
    # one case per (word, sense): scores of the word's other sentences + true yes/no
    rng = np.random.default_rng(seed)
    cases = []
    for word in words:
        rows = np.where(df["lemma"].values == word)[0]
        senses = df["sense_id"].values[rows]
        for sense in np.unique(senses):
            sense_rows = rows[senses == sense]
            if len(sense_rows) <= K:
                continue
            examples = rng.choice(sense_rows, K, replace=False)
            reference = normalize(vectors[examples].mean(axis=0))
            others = np.setdiff1d(rows, examples)
            cases.append({
                "word": word,
                "scores": vectors[others] @ reference,
                "labels": senses[np.searchsorted(rows, others)] == sense,
            })
    return cases


def f1(cases, cutoff):
    # F1 per case, averaged per word, then across words
    rows = [{"word": c["word"], "f1": f1_score(c["labels"], c["scores"] >= cutoff, zero_division=0)} for c in cases]
    return pd.DataFrame(rows).groupby("word")["f1"].mean().mean()


def train(df, vectors, split):
    vectors = normalize(vectors.astype(np.float32))
    train_words = split.loc[split["split"] == "train", "lemma"]
    dev_words = split.loc[split["split"] == "dev", "lemma"]

    train_cases = make_cases(df, vectors, train_words)
    cutoffs = np.round(np.arange(0, 1.01, 0.01), 2)
    best = max(cutoffs, key=lambda c: f1(train_cases, c))

    dev_f1 = f1(make_cases(df, vectors, dev_words), best)
    print(f"best cutoff (train): {best} | train F1 = {f1(train_cases, best):.3f} | dev F1 = {dev_f1:.3f}")

    os.makedirs("models", exist_ok=True)
    json.dump({"cutoff": float(best), "k": K}, open(MODEL_FILE, "w"))
    return best


def predict(examples, sentences):
    """examples / sentences: lists of (text, start, end).
    examples = known uses of the target sense, sentences = new uses to label."""
    from WordTransformer import WordTransformer, InputExample

    cutoff = json.load(open(MODEL_FILE))["cutoff"]
    model = WordTransformer("pierluigic/xl-lexeme")
    encode = lambda items: normalize(model.encode([InputExample(texts=t, positions=[s, e]) for t, s, e in items]))

    reference = normalize(encode(examples).mean(axis=0))
    scores = encode(sentences) @ reference
    return [(text, round(float(score), 3), bool(score >= cutoff)) for (text, _, _), score in zip(sentences, scores)]
