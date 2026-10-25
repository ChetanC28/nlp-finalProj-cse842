# Runs the whole pipeline. Run from the repo root: python main.py
import os

import numpy as np
import pandas as pd

from src.load_data import load_sentences
from src.clean_data import clean_sentences
from src.split import split_words
from src.subtask1 import run_subtask1
from src.embed import embed_func
from src import baseline

# Step 1: load the data
df = load_sentences()
print(len(df), "sentences")

# Step 2: clean
df = clean_sentences(df)

# Step 3: split words into train / dev / test (80 / 10 / 10)
split = split_words(df)
print(split["split"].value_counts().to_dict())

# Step 4: load the vectors (made by the encoding step)
if not os.path.exists("cache/embeddings.npy"):
    print("No vectors yet: Calculating the embeddings and putting them in cache/")
    embed_func()

df = pd.read_parquet("cache/embeddings_sentences.parquet")
vectors = np.load("cache/embeddings.npy")

# Step 5: Subtask 1 (unsupervised sense induction & change detection)
run_subtask1(df, vectors, split)

# Step 6: Subtask 2 basic model (prototype + one cutoff): train on train words, score on dev and test
# cutoff = baseline.train(df, vectors, split)
# baseline.evaluate(df, vectors, split, "dev", cutoff)
# baseline.evaluate(df, vectors, split, "test", cutoff)
