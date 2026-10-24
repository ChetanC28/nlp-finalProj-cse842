# Runs the whole pipeline. Run from the repo root: python main.py
import os

import numpy as np
import pandas as pd

from src.load_data import load_sentences
from src.clean_data import clean_sentences
from src.split import split_words
from src.subtask2 import run_subtask2

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
    print("No vectors yet: put embeddings.npy and embeddings_sentences.parquet in cache/")
    raise SystemExit

df = pd.read_parquet("cache/embeddings_sentences.parquet")
vectors = np.load("cache/embeddings.npy")

# Step 5: Subtask 2
run_subtask2(df, vectors, split)
