# Turn each sentence into an XL-LEXEME vector for its target word, and save them.
# Optionally fine-tune XL-LEXEME on our train words first (no year: the model only sees text + word position).
#
# Pretrained:  embed_func()                -> cache/embeddings.npy, cache/embeddings_sentences.parquet
# Fine-tuned:  finetune(); embed_func(finetuned=True)
#                                          -> models/xl-lexeme-ft/, cache/embeddings_ft.npy, cache/embeddings_ft_sentences.parquet
# Both at once, from the repo root: python -m src.embed
import os
import shutil

import numpy as np
import pandas as pd
import torch
from WordTransformer import InputExample, WordTransformer

MODEL_NAME = "pierluigic/xl-lexeme"
FINETUNED_MODEL = "models/xl-lexeme-ft"
FINETUNE_SPLIT_FILE = "splits/finetune_split.csv"

# fine-tuning settings (see DECISIONS.md, Fine-tuning)
HOLDOUT_SHARE = 0.2    # share of train words kept out of fine-tuning, used to pick cutoffs
PAIRS_PER_WORD = 50    # half same sense, half different sense of the same word
EPOCHS = 1
LEARNING_RATE = 1e-5
BATCH_SIZE = 16        # pairs per step
WARMUP_SHARE = 0.1     # share of steps with a rising learning rate
SEED = 42


def pick_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_model(model_name=MODEL_NAME):
    device = pick_device()
    print(device)
    return WordTransformer(model_name, device=device)


def save_npy(path, array):
    # write to a temp file first, so a crash mid-write never leaves a half-saved file
    with open(path + ".tmp", "wb") as f:
        np.save(f, array)
    os.replace(path + ".tmp", path)


def embed_sentences(df, model=None, batch_size=32, name="embeddings", chunk_size=50_000, model_name=MODEL_NAME):
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
            model = load_model(model_name)
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


def cleaned_sentences():
    from src.clean_data import clean_sentences
    return clean_sentences(pd.read_parquet("cache/sentences.parquet"))


def embed_func(sample_size=None, finetuned=False):
    df = cleaned_sentences()
    name = "embeddings_ft" if finetuned else "embeddings"
    model_name = FINETUNED_MODEL if finetuned else MODEL_NAME
    if sample_size is not None:
        df = df.sample(sample_size, random_state=0).reset_index(drop=True)
        name += "_sample"
    vectors = embed_sentences(df, name=name, model_name=model_name)
    print(vectors.shape, f"vectors saved to cache/{name}.npy")
    return vectors


# --- Fine-tuning ---

def finetune_split(seed=SEED):
    """Split the train words into "finetune" (80%) and "holdout" (20%). dev/test words are not touched.
    Saved to splits/finetune_split.csv so the whole team uses the same one."""
    if os.path.exists(FINETUNE_SPLIT_FILE):
        return pd.read_csv(FINETUNE_SPLIT_FILE)

    split = pd.read_csv("splits/word_split.csv")
    words = np.array(sorted(split.loc[split["split"] == "train", "lemma"]))
    np.random.default_rng(seed).shuffle(words)

    n_holdout = int(HOLDOUT_SHARE * len(words))
    part = ["holdout"] * n_holdout + ["finetune"] * (len(words) - n_holdout)
    out = pd.DataFrame({"lemma": words, "part": part})
    out.to_csv(FINETUNE_SPLIT_FILE, index=False)
    return out


def make_pairs(df, words, seed=SEED):
    """Per word: PAIRS_PER_WORD / 2 same-sense pairs (label 1) and as many different-sense pairs (label 0),
    both sentences of the same word. Words with only one sense or no sense with 2+ sentences are skipped,
    so every word stays 50/50. Only text and word position go in, no year."""
    rng = np.random.default_rng(seed)
    pairs = []
    for word in words:
        rows = np.where(df["lemma"].values == word)[0]
        senses = df["sense_id"].values[rows]
        names, counts = np.unique(senses, return_counts=True)
        repeated = rows[np.isin(senses, names[counts >= 2])]  # sentences whose sense has a 2nd sentence
        if len(names) < 2 or len(repeated) == 0:
            continue

        for _ in range(PAIRS_PER_WORD // 2):
            a = rng.choice(repeated)
            same = rows[(senses == df["sense_id"].values[a]) & (rows != a)]
            pairs.append(make_pair(df, a, rng.choice(same), 1))

            a = rng.choice(rows)
            other = rows[senses != df["sense_id"].values[a]]
            pairs.append(make_pair(df, a, rng.choice(other), 0))
    return pairs


def make_pair(df, a, b, label):
    row_a, row_b = df.iloc[a], df.iloc[b]
    return InputExample(
        texts=[row_a["text"], row_b["text"]],
        positions=[[int(row_a["start"]), int(row_a["end"])], [int(row_b["start"]), int(row_b["end"])]],
        label=label,
    )


def finetune(df=None):
    """Fine-tune XL-LEXEME on the "finetune" train words with a contrastive loss, save to models/xl-lexeme-ft/."""
    from sentence_transformers import losses
    from torch.utils.data import DataLoader

    if df is None:
        df = cleaned_sentences()
    parts = finetune_split()
    words = parts.loc[parts["part"] == "finetune", "lemma"]
    pairs = make_pairs(df, words)
    print(f"{len(pairs)} training pairs from {len(words)} words")

    torch.manual_seed(SEED)
    model = load_model()
    loader = DataLoader(pairs, shuffle=True, batch_size=BATCH_SIZE)
    # cosine distance; same sense pulled together, different sense pushed beyond the margin (default 0.5)
    loss = losses.ContrastiveLoss(model)
    steps = len(loader) * EPOCHS

    os.makedirs("models", exist_ok=True)
    model.fit(
        train_objectives=[(loader, loss)],
        epochs=EPOCHS,
        warmup_steps=int(WARMUP_SHARE * steps),
        optimizer_params={"lr": LEARNING_RATE},
        use_amp=True,  # mixed precision, to fit in 16 GB
        output_path=FINETUNED_MODEL,
    )
    return model


if __name__ == "__main__":
    df = cleaned_sentences()
    finetune(df)
    embed_func(finetuned=True)
