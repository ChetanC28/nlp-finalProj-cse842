# Runs the whole pipeline. Run from the repo root: python main.py
import os

import numpy as np
import pandas as pd

from src.load_data import load_sentences
from src.clean_data import clean_sentences
from src.split import split_words
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
    print("No vectors yet: put embeddings.npy and embeddings_sentences.parquet in cache/")
    raise SystemExit

df = pd.read_parquet("cache/embeddings_sentences.parquet")
vectors = np.load("cache/embeddings.npy")

# Step 5: train the basic Subtask 2 model (prototype + one cutoff)
baseline.train(df, vectors, split)

# Step 6: try it on an unseen input
examples = [  # known uses of the target sense ("boss" = cow)
    ("The boss or ox when called will run to his owner.", 4, 8),
    ("Call the boss in from the field before dark.", 9, 13),
    ("The farmer milked the boss every morning.", 22, 26),
]
sentences = [  # new uses to label
    ("The boss wandered into the barn to be milked.", 4, 8),
    ("My boss gave me a raise last week.", 3, 7),
    ("The shield had a metal boss in the center.", 23, 27),
]
for text, score, answer in baseline.predict(examples, sentences):
    print(f"{'YES' if answer else 'NO ':3}  {score}  {text}")
