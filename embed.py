# Turn each sentence into an XL-LEXEME vector for its target word, and save them.
import os
import shutil

import numpy as np
import torch
from WordTransformer import InputExample, WordTransformer

MODEL_NAME = "pierluigic/xl-lexeme"


def pick_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_model():
    device = pick_device()
    print(device)
    return WordTransformer(MODEL_NAME, device=device)


def save_npy(path, array):
    # write to a temp file first, so a crash mid-write never leaves a half-saved file
    with open(path + ".tmp", "wb") as f:
        np.save(f, array)
    os.replace(path + ".tmp", path)


def embed_sentences(df, model=None, batch_size=32, name="embeddings", chunk_size=50_000):
    # XL-LEXEME marks the word at text[start:end], so run clean_sentences first.
    # Saves cache/<name>.npy and cache/<name>_sentences.parquet.
    # Each chunk of rows is saved as it finishes, so a crashed run picks up where it stopped.
    # The chunks are only valid for the same df, so delete cache/<name>_chunks/ if the cleaning changes.
    chunk_dir = f"cache/{name}_chunks"
    os.makedirs(chunk_dir, exist_ok=True)

    chunks = []
    for i, first in enumerate(range(0, len(df), chunk_size)):
        path = f"{chunk_dir}/{i:04d}.npy"
        if os.path.exists(path):
            chunks.append(np.load(path))
            continue

        if model is None:
            model = load_model()
        rows = df.iloc[first:first + chunk_size]
        print(f"chunk {i}: rows {first} to {first + len(rows)} of {len(df)}")
        examples = [
            InputExample(texts=text, positions=[start, end])
            for text, start, end in zip(rows["text"], rows["start"], rows["end"])
        ]
        vectors = model.encode(examples, batch_size=batch_size, show_progress_bar=True)

        # float16 halves the file size (~1.3 GB for all rows)
        vectors = np.asarray(vectors, dtype=np.float16)
        save_npy(path, vectors)
        chunks.append(vectors)

    # row i of the array is row i of df
    vectors = np.concatenate(chunks)
    save_npy(f"cache/{name}.npy", vectors)
    df.to_parquet(f"cache/{name}_sentences.parquet")
    shutil.rmtree(chunk_dir)
    return vectors


# Run from the repo root (after main.py has made the cache):
#   python embed.py         embeds every sentence (hours on CPU) -> cache/embeddings.npy
#   python embed.py 1000    embeds a random 1000 to test -> cache/embeddings_sample.npy
# If a run stops partway, run the same command again to continue.
if __name__ == "__main__":
    import sys

    import pandas as pd
    from src.clean_data import clean_sentences

    df = clean_sentences(pd.read_parquet("cache/sentences.parquet"))
    name = "embeddings"
    if len(sys.argv) > 1:
        df = df.sample(int(sys.argv[1]), random_state=0).reset_index(drop=True)
        name = "embeddings_sample"
    vectors = embed_sentences(df, name=name)
    print(vectors.shape, f"vectors saved to cache/{name}.npy")
