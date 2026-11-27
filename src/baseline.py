# Most basic Subtask 2 model: prototype + one fixed cutoff. No year information.
#
# train():    find the cutoff that gives the best F1 on the train words, save it.
# evaluate(): score the model on a set of words (dev or test).
# predict():  for a new input (example sentences of a sense + new sentences), say yes/no.
import json
import os

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

K = 3  # number of example sentences per sense (Subtask 2 decision 1)
MODEL_FILE = "models/baseline.json"        # cutoff learned on the pretrained vectors
FT_MODEL_FILE = "models/baseline_ft.json"  # cutoff learned on the fine-tuned vectors


def model_file(finetuned=False):
    return FT_MODEL_FILE if finetuned else MODEL_FILE


def tuning_words(split, finetuned=False):
    """Words used to pick settings (cutoff, smoothing).
    Pretrained vectors: all train words. Fine-tuned vectors: only the held-out train words the model
    was NOT trained on (splits/finetune_split.csv); on the others its scores look better than they are."""
    if finetuned:
        from src.embed import FINETUNE_SPLIT_FILE
        parts = pd.read_csv(FINETUNE_SPLIT_FILE)
        return parts.loc[parts["part"] == "holdout", "lemma"]
    return split.loc[split["split"] == "train", "lemma"]


def normalize(v):
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def make_cases(df, vectors, words, seed=0):
    # one case per (word, sense): scores of the word's other sentences + truth labels.
    # truth: YES if the sentence's sense_id is the target sense, NO otherwise.
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


def metrics(cases, cutoff):
    # each metric per case, averaged per word, then across words
    rows = []
    for c in cases:
        pred = c["scores"] >= cutoff
        rows.append({
            "word": c["word"],
            "f1": f1_score(c["labels"], pred, zero_division=0),
            "precision": precision_score(c["labels"], pred, zero_division=0),
            "recall": recall_score(c["labels"], pred, zero_division=0),
            "accuracy": accuracy_score(c["labels"], pred),
        })
    return pd.DataFrame(rows).groupby("word").mean().mean().round(3).to_dict()


def train(df, vectors, split, finetuned=False):
    """finetuned=True: the vectors come from the fine-tuned model, so the cutoff is picked on the holdout words."""
    vectors = normalize(vectors.astype(np.float32))
    train_words = tuning_words(split, finetuned)

    cases = make_cases(df, vectors, train_words)
    cutoffs = np.round(np.arange(0, 1.01, 0.01), 2)
    best = max(cutoffs, key=lambda c: metrics(cases, c)["f1"])
    print(f"best cutoff ({'holdout' if finetuned else 'train'}): {best} | {metrics(cases, best)}")

    os.makedirs("models", exist_ok=True)
    json.dump({"cutoff": float(best), "k": K}, open(model_file(finetuned), "w"))
    return best


def evaluate(df, vectors, split, name, cutoff):
    vectors = normalize(vectors.astype(np.float32))
    words = split.loc[split["split"] == name, "lemma"]
    result = metrics(make_cases(df, vectors, words), cutoff)
    print(f"{name}: {result}")
    return result


def predict(examples, sentences, finetuned=False):
    """examples / sentences: lists of (text, start, end).
    examples = known uses of the target sense, sentences = new uses to label."""
    from WordTransformer import WordTransformer, InputExample
    from src.embed import FINETUNED_MODEL, MODEL_NAME

    cutoff = json.load(open(model_file(finetuned)))["cutoff"]
    model = WordTransformer(FINETUNED_MODEL if finetuned else MODEL_NAME)
    encode = lambda items: normalize(model.encode([InputExample(texts=t, positions=[s, e]) for t, s, e in items]))

    reference = normalize(encode(examples).mean(axis=0))
    scores = encode(sentences) @ reference
    return [(text, round(float(score), 3), bool(score >= cutoff)) for (text, _, _), score in zip(sentences, scores)]
