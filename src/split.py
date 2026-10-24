# Split the words (not sentences) into train / dev / test = 80 / 10 / 10.
# The split is saved to splits/word_split.csv so the whole team uses the same one.
import os

import numpy as np
import pandas as pd

SPLIT_FILE = "splits/word_split.csv"


def split_words(df, seed=42):
    if os.path.exists(SPLIT_FILE):
        return pd.read_csv(SPLIT_FILE)

    words = np.array(sorted(df["lemma"].unique()))
    np.random.default_rng(seed).shuffle(words)

    n_train = int(0.8 * len(words))
    n_dev = int(0.1 * len(words))
    split = ["train"] * n_train + ["dev"] * n_dev + ["test"] * (len(words) - n_train - n_dev)

    out = pd.DataFrame({"lemma": words, "split": split})
    os.makedirs("splits", exist_ok=True)
    out.to_csv(SPLIT_FILE, index=False)
    return out
