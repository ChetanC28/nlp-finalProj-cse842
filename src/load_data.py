# Load all sentences into one table and save it.
import glob
import os

import pandas as pd


def load_sentences():
    files = glob.glob("NLP Dataset/generated_sentences/*/*.jsonl")
    df = pd.concat([pd.read_json(f, lines=True) for f in files], ignore_index=True)

    os.makedirs("cache", exist_ok=True)
    df.to_parquet("cache/sentences.parquet")
    return df
