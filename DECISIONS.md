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

## Split (`src/split.py`)

- Split by **word**, not sentence: 80 / 10 / 10 → 1,874 train, 234 dev, 235 test words (seed 42).
- Saved to `splits/word_split.csv` and committed, so everyone uses the same split.
- **train**: pick settings (cutoff, smoothing). **dev**: compare ideas (e.g. extra cleaning). **test**: final number only.

## Subtask 1a baseline (`src/subtask1.py`)

Unsupervised sense induction using Agglomerative Hierarchical Clustering (AHC) with average linkage and cosine distance (threshold = 0.35). Evaluated using BCubed F1 against gold senses.

- **Baseline test (5 dev words, 1,437 sentences)**: Macro BCubed F1 = 0.617 (Precision = 0.637, Recall = 0.663).
- **Ideas to explore beyond baseline**:
  - **Temporal-aware clustering**: Incorporate publication periods as a prior or constraint so senses evolve chronologically rather than pooling all years together.
  - **Dynamic thresholds**: Instead of a fixed 0.35 threshold for all words, calibrate thresholds per word using silhouette analysis or density estimation.

## Subtask 2 (`src/subtask2.py`)

Skeleton only, decisions pending. See the DECISION comments in `src/subtask2.py` (12 open decisions).

Input: `cache/embeddings.npy` + `cache/embeddings_sentences.parquet` (same row order), from the encoding step.

Agreed so far:
- Each sense is represented by a prototype: the average XL-LEXEME vector of k example sentences (Cassotti & Tahmasebi, 2025; Snell et al., 2017).

## Subtask 2 baseline (`src/baseline.py`)

Simplest working version, no year information. This is the number our temporal method has to beat.

- **train:** for each (word, sense) in the train words, average k example vectors → reference; score the word's other sentences by cosine similarity; pick the one cutoff with the best F1. Saved to `models/baseline.json`.
- **evaluate:** same on dev / test words with the learned cutoff. Truth labels come from `sense_id`: YES if the sentence's sense is the target sense, NO otherwise.
- **predict:** given example sentences of a sense + new sentences (text, start, end), encode with XL-LEXEME and answer yes/no.
- Placeholders (not decided yet): k = 3; cutoffs tried 0.00–1.00 in steps of 0.01.
- Encoding speed on a MacBook (MPS): ~93 sentences/s → full dataset ≈ 2 hours.

**Results (full dataset, real XL-LEXEME vectors, cutoff = 0.74):**

| | F1 | Precision | Recall | Accuracy |
|---|---|---|---|---|
| train | 0.537 | 0.496 | 0.806 | 0.685 |
| dev | 0.528 | 0.482 | 0.806 | 0.673 |
| test | **0.536** | 0.498 | 0.790 | 0.680 |

**What it tells us:**
- F1 ≈ 0.53: the model finds a real signal but is far from perfect.
- High recall, low precision: it finds ~80% of the target sense's sentences, but about half its YES answers are wrong (says yes too often). Likely cause: senses of the same word that are close in meaning get similar scores (e.g. `adversarial` general vs. legal).
- Train ≈ dev ≈ test: only one setting (the cutoff), so no overfitting; the number carries over to unseen words.
- Our temporal method targets exactly these borderline false YES answers, so if it works, **precision should go up**.

**Coverage:** every dev/test word and sentence takes part, but senses with ≤ k sentences are never the target (128 of 1,239 dev senses, 123 of 1,187 test senses ≈ 10%). These are the rarest senses. Their sentences are still scored as NO cases for other senses. The k example sentences of each case are not scored.

**Noticed, not changed:**
-  most answers are NO, so accuracy looks better than the model is. Report F1 as the main metric.


## For the report

Explanations to include:
- Gold labels are the senses each sentence was **generated** to exemplify (Saponara et al., 2026), not human-checked, so some label noise is expected (e.g. `bust` → "burst"). This is why we report improvement over the baseline rather than absolute scores (proposal §4.1).
- Truth for Subtask 2 is built from `sense_id`: target sense = YES, any other sense of the same word = NO.
- Split by word, so test words are never seen during tuning.
- Senses with ≤ k sentences were not evaluated as targets (~10% of senses, the rarest ones).
- Data cleaning: 3.4% of sentences dropped, 1.2% fixed (see Cleaning).

Open questions to answer (and say in the report):
- What k (number of example sentences) and why? Trade-off: larger k = better reference but more rare senses skipped.
- Does the year information raise precision without lowering recall?
- Does it help or hurt on stable vs. newly appearing vs. rare senses? (promised in the proposal)
- Since sentence counts per period are roughly equal (see Data), can trajectories from our predictions show how common a sense is, or only whether it exists?
- Do the 29 sentences with two different labels (e.g. "Bots.") noticeably lower the score?
- Is one random pick of example sentences stable, or do scores change with a different pick?
