# Decisions Log

What we did, why, and what we noticed but didn't change. Update as we go.

## Setup

- Python 3.11, all libraries pinned in `requirements.txt`, installed with `./setup.sh`.
- XL-LEXEME pinned to one commit of [pierluigic/xl-lexeme](https://github.com/pierluigic/xl-lexeme).

## Data

- Dataset: Saponara et al. (2026), synthetic English, 1810–1990 in 10 periods.
- 672,120 sentences, 2,344 words, 11,707 senses.
- Each `.jsonl` file = one word; each line = one sentence with `lemma`, `sense_id`, `year`, `start`, `end`, `text`.

**Noticed, not changed:**
- Sentence counts are roughly equal per period (~10 per sense), **not** proportional to the true sense frequency in `metadata/`. So trajectories built from our predictions mostly show whether a sense exists in a period, not how common it is.
- 2,346 files but 2,344 words with sentences: `honk.jsonl` and `pinheaded.jsonl` are empty (0 sentences).
- `Jack` (27 senses, proper noun) and `jack` (12 senses) are separate words in separate files; kept separate.

## Loading (`src/load_data.py`)

- All files combined into one pandas table, saved to `cache/sentences.parquet`.

## Cleaning (`src/clean_data.py`)

Checks whether `text[start:end]` (the "pointer") is the target word.

| Case | Rule | Rows | Example |
|---|---|---|---|
| Keep | Pointer's first 3 letters match the word (ignoring hyphens/spaces) | ~641k | `ship` → "shipped", `forcible` → "forecible" |
| Fix | Pointer is wrong, but the word is in the sentence → move the pointer | 8,099 | `control`: "a control" → "control" |
| Drop | Word is not in the sentence at all | 23,055 | `bust`: sentence has "burst" |

Result: 672,120 → **649,065 sentences** (3.4% dropped).

Why: XL-LEXEME marks the word at `start:end`, so a wrong pointer gives a wrong vector. Inflected forms are fine. Approach follows WiC (Pilehvar & Camacho-Collados, 2019): fix errors when possible, remove otherwise.

**Noticed, not changed:**
- Some real spelling variants get dropped because their first 3 letters differ (`icily` → "Icy", `eye` → "Eying"). Small number; accepted.
- When fixing, we take the first match in the sentence, which is occasionally wrong (`go` matched "got" instead of "gō").
- Duplicates (not removed yet):
  - 57 fully identical rows, e.g. `French`: "To French a Leg of Mutton." (1810) appears twice.
  - 213 sentences reused across years with the same sense, e.g. "The Anglo-Roman Church." in 1810, 1870, 1890, 1910, 1930. Harmless.
  - 29 sentences labeled with **two different senses**, e.g. `bot`: "Bots." is both `bot_nn01-16381146` and `bot_nn01-16381258`. No context to tell them apart, so no model can get these right.
