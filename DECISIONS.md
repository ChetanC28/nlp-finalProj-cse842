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
- Sentence counts per (sense, period) follow the true frequency in `metadata/` in **order** (Spearman 0.89) but are heavily **compressed**: on average roughly log₂(frequency), noisy, capped at 30. E.g. frequency 100–1,000 → ~8 sentences, 10,000+ → ~21. Frequency 0 → almost always no sentences, but 78 sense-period pairs (997 sentences) have sentences where the metadata says frequency 0 (e.g. `paragraph_vb01-31872348` in 1810–1850), so data and metadata don't fully agree. So trajectories from our predictions do show how common a sense is, but on a log-like scale: a 100× change looks like ~2–3×. (Our guess at why: proportional sampling would leave rare senses with almost no sentences. Check the dataset paper.)
- 2,346 files but 2,344 words with sentences: `honk.jsonl` and `pinheaded.jsonl` are empty (0 sentences).
- `Jack` (proper noun, in folder `27_senses` but only 26 senses have sentences) and `jack` (12 senses) are separate words in separate files; kept separate.

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
  - 192 sentence texts reused across years with the same sense, e.g. "The Anglo-Roman Church." in 1810, 1870, 1890, 1910, 1930. Harmless. (21 more repeat within the same year with different start/end.)
  - 29 sentence texts (62 rows) labeled with **two different senses**, e.g. `bot`: "Bots." is both `bot_nn01-16381146` and `bot_nn01-16381258`. No context to tell them apart, so no model can get these right.
  - All counts above are on the raw data, before cleaning.

## Split (`src/split.py`)

- Split by **word**, not sentence: 80 / 10 / 10 → 1,874 train, 234 dev, 235 test words (seed 42).
- Saved to `splits/word_split.csv` and committed, so everyone uses the same split.
- **train**: pick settings (cutoff, smoothing). **dev**: compare ideas (e.g. extra cleaning). **test**: final number only.

## Subtask 1 baseline (`src/subtask1.py`)

Unsupervised sense induction (1a) and diachronic sense dynamics tracking (1b).

- **Subtask 1a (Diachronic Sense Assignment)**: Agglomerative Hierarchical Clustering (AHC) with average linkage and cosine distance (threshold = 0.35). Evaluated using Macro BCubed F1 against gold senses.
- **Subtask 1b (Sense Dynamics)**: Step-by-step Jensen-Shannon Divergence (JSD) across consecutive time periods. Evaluated using Spearman's rank correlation ($\rho$) against gold semantic change rankings.

**Results (full dataset, 234 dev words, real XL-LEXEME vectors):**

| Track | Metric | Score | Details |
|---|---|---|---|
| **Subtask 1a** | Macro BCubed F1 | **0.627** | Unsupervised sense clustering across all periods |
| **Subtask 1b** | Spearman $\rho$ | **0.400** | Rank correlation of temporal JSD change scores |

**Decisions & Edge Cases Handled:**
- **Missing Periods**: 35 dev words (e.g. *boast*, *downsize*) do not appear in all 10 periods. Sense distributions and consecutive JSD are computed strictly over observed periods (`total_in_period > 0`) to prevent zero-division / NaN values. Words appearing in $\le 1$ period are assigned 0.0 change.
- **Plot Generation**: Automatically outputs two evaluation figures to `plots/`:
  - `subtask1a_bcubed_distribution.png`: Histogram showing the distribution of BCubed F1 scores across dev words with the macro mean line.
  - `subtask1b_change_correlation.png`: Scatter plot of `true_changes` vs. `predicted_changes` with Spearman $\rho$.
- **Notable Observations**: Highly dynamic polysemous words (e.g. *ruck*, *hood*) with multiple evolving senses accumulated the highest change in both gold annotations and predicted clusters.

**Ideas to explore beyond baseline**:
- **Temporal-aware clustering**: Incorporate publication periods as a prior or constraint so senses evolve chronologically rather than pooling all years together.
- **Dynamic thresholds**: Instead of a fixed 0.35 threshold for all words, calibrate thresholds per word using silhouette analysis or density estimation.

## Subtask 2 (`src/subtask2.py`)

Our method: the baseline, plus **one cutoff per period** that comes from a smoothed sense trajectory. Implemented; dev results below (see Results).

Input: `cache/embeddings.npy` + `cache/embeddings_sentences.parquet` (same row order), from the encoding step. Run: `python -m src.subtask2`. Saves the tuned settings to `models/temporal.json`.

**How it works (per word and sense):**
1. Reference = average of k example vectors; score the word's other sentences by cosine similarity (same as the baseline).
2. First pass: baseline answers with its one cutoff.
3. Trajectory: for each period, the share of the word's sentences that the first pass says YES to.
4. Smooth the 10-point trajectory.
5. Per-period cutoff: lower where the sense is more common than its own average, higher where it is rarer.
6. Final answer: score ≥ the cutoff of the sentence's own period.

**Where the year comes in:** only the `year` column of each sentence (10 values, one per period). It is used in steps 3 and 6. The embeddings carry no year information.

**Decisions:**

| # | Decision | Choice | Why |
|---|---|---|---|
| — | Put the year into the embeddings? | **No.** The year is used only at the decision step. (A fine-tuned model **without** the year is added as a second set of vectors, see Fine-tuning.) | Baseline and temporal method use the **same scores**, so any difference comes only from the year. Year tokens alone (no training) would make same-period sentences look alike whatever their sense → more false YES. Fine-tuning **with** the year (e.g. time masking, Rosin et al., 2022) would also need a no-year control model → future work. |
| 1 | k (example sentences) | **3**, same as baseline | Must match the baseline for a fair comparison. Best k is a separate question for both models (see For the report). |
| 2 | Where examples come from | **Random, any period**, same as baseline | Earliest-period-only would give no reference for senses that only appear later, which is one of the groups we report on. |
| 3 | Senses with ≤ k sentences | **Skipped as targets**, same as baseline | Otherwise nothing of the sense is left to score. ~10% of senses; their sentences still count as NO cases. |
| 4 | Which sentences are scored | **All of the word's sentences except the k examples**, same as baseline | Same-word, other-sense sentences are realistic NO cases. Same seed (0) → identical examples and scores to the baseline (checked). |
| 5 | Trajectory measure | **Share** of the word's sentences in the period that the first pass says YES to. Periods without sentences are left out (not 0%). | A raw count mixes "sense is common" with "word has many sentences in this period". First-pass precision is ~50%, so shares are too high everywhere, but step 5 only uses rises and falls, not the level. |
| 6 | Smoothing method | **Gaussian kernel** over periods, strength σ in periods (0 = none). Only observed periods are averaged; weights renormalized. | One continuous setting that can reach beyond direct neighbours. A 3-point moving average can't; a spline is too much for 10 points. |
| 7 | Trajectory → cutoffs | `cutoff_t = base_cutoff − adjust × (smoothed_t − sense average)` | Linear, two settings (σ, adjust). `adjust = 0` is exactly the baseline. The log-ratio prior used for Subtask 1 needs a pseudocount for absent periods (a third setting), and that temporal attempt hurt more the more it changed, so we keep the adjustment simple and small. |
| 8 | Averaging | Per (word, sense) case → per word → across words, same as `baseline.metrics` | Same as baseline and proposal ("per word, averaged across words"). |
| 9 | Stable / new / rare breakdown | **Later**, after the main method works | Needs its own definitions from `metadata/`. |
| 10 | Values tried | σ ∈ {0, 0.5, 1, 2, 3}; adjust ∈ {0, 0.05, 0.1, 0.2, 0.3, 0.5} | 30 combinations; includes the baseline (σ = 0 or adjust = 0). |
| 11 | Tuning order | **Sequential**: base cutoff = the baseline's learned cutoff (0.74, `models/baseline.json`); then grid σ × adjust by train F1. Ties → least smoothing / smallest adjust. | Fewer combinations, less room to overfit train, and the first pass is exactly the baseline. |
| 12 | Random pick of examples | **Once, seed 0** | Same pick as the baseline, so both are compared on identical references. Stability across seeds is a later check (see For the report). |

**What the code does for each new step** (the steps the baseline doesn't have; reasoning in the table above):

- **Sense trajectory** (`sense_trajectory()`, decision 5)
  - Input: the first-pass YES/NO answers of one case and the period of each scored sentence.
  - For each of the 10 periods: YES answers ÷ the word's scored sentences in that period, i.e. a **share**, not a raw count. Corrects for periods having different numbers of sentences.
  - A period where the word has no scored sentences becomes NaN (missing), not 0%. The k example sentences are not counted.
  - No true labels are used; only the baseline's answers.
- **Smoothing** (`smooth()`, decision 6)
  - **Gaussian kernel** over the 10 periods: each period's smoothed share is a weighted average of all observed periods, with weights falling off with distance in time.
  - Strength = σ in periods. σ = 0 returns the trajectory unchanged (no smoothing); σ ≈ 1 mixes in mostly the direct neighbours; larger σ flattens more.
  - Missing periods are left out of the average and weights are renormalized, so the two ends and gaps are handled; missing periods stay NaN.
- **Cutoff per period** (`period_cutoffs()`, `predict_temporal()`, decision 7)
  - Rule: `cutoff_t = base_cutoff − adjust × (smoothed_t − sense average)`. Sense average = mean of the smoothed shares over observed periods (each period counts the same).
  - Sense more common than its average → lower cutoff (borderline sentences lean YES); rarer → higher cutoff (lean NO).
  - `adjust` sets how far the cutoff moves. `adjust = 0` gives exactly the baseline's answers (checked).
  - Each sentence is compared with the cutoff of **its own period**. This is the only place the year changes an answer.
- **Tuning** (`tune()`, `run_subtask2()`, decisions 10–11)
  - Base cutoff **kept fixed** at the baseline's learned cutoff (`models/baseline.json`, 0.74); only σ and adjust are tuned. So the first pass is exactly the baseline, and only the time-based part differs.
  - Grid: σ ∈ {0, 0.5, 1, 2, 3} × adjust ∈ {0, 0.05, 0.1, 0.2, 0.3, 0.5} = 30 combinations, scored by F1 on the tuning words. Ties → the first one, i.e. least smoothing / smallest adjust.
  - Tuning words: all train words for the pretrained vectors; only the 374 holdout words for the fine-tuned vectors (`baseline.tuning_words()`, see Fine-tuning).
  - Chosen settings saved to `models/temporal.json` (pretrained) or `models/temporal_ft.json` (fine-tuned). Pretrained run picked **σ = 1, adjust = 0.3**.

**Results (dev, 234 words, 1,111 word–sense cases):**

| Vectors | Method | Settings | F1 | Precision | Recall | Accuracy |
|---|---|---|---|---|---|---|
| pretrained | baseline | cutoff 0.74 | 0.528 | 0.482 | 0.806 | 0.673 |
| pretrained | temporal | cutoff 0.74, σ = 1, adjust = 0.3 | 0.530 | 0.483 | 0.809 | 0.675 |
| fine-tuned | baseline | cutoff 0.80 | 0.576 | 0.533 | 0.820 | 0.732 |
| fine-tuned | temporal | cutoff 0.80, σ = 2, adjust = 0.3 | **0.578** | 0.534 | 0.822 | 0.734 |

**Results (test, 235 words, 1,064 word–sense cases; final numbers, settings frozen):**

| Vectors | Method | Settings | F1 | Precision | Recall | Accuracy |
|---|---|---|---|---|---|---|
| pretrained | baseline | cutoff 0.74 | 0.536 | 0.498 | 0.790 | 0.680 |
| pretrained | temporal | cutoff 0.74, σ = 1, adjust = 0.3 | 0.538 | 0.498 | 0.793 | 0.682 |
| fine-tuned | baseline | cutoff 0.80 | 0.584 | 0.546 | 0.801 | 0.742 |
| fine-tuned | temporal | cutoff 0.80, σ = 2, adjust = 0.3 | **0.586** | 0.548 | 0.805 | 0.743 |

Test was scored once with the saved settings (no tuning on test). Pretrained baseline matches the earlier baseline test run exactly. **From here on, settings must not be changed based on test**; new ideas (oracle check, definition variant, stable/new breakdown) are judged on dev.

On the tuning words (settings picked here): pretrained = all 1,874 train words, baseline 0.537 → temporal 0.539 F1; fine-tuned = 374 holdout words, baseline 0.592 → temporal 0.594 F1.

Settings from `models/baseline.json`, `models/temporal.json`, `models/baseline_ft.json`, `models/temporal_ft.json`. The scripts only print scores, so these were recomputed from the saved settings (pretrained numbers identical to the original run).

**What it tells us:**
- **Fine-tuning is the big gain:** baseline dev F1 0.528 → 0.576 (+0.048), mostly from precision (+0.051; recall +0.014). The model says YES wrongly less often, which was the baseline's main weakness.
- **The temporal method adds only +0.002 F1**, on both vector sets. The gain comes mostly from recall (+0.002–0.003); precision barely moves (+0.001), so it does **not** yet do what we hoped (fewer false YES). Not tested for significance; a difference this small may be noise.
- **Better vectors did not make the temporal step help more** (+0.002 either way), although their first pass is more accurate. Suggests the limit is the per-period cutoff idea on this data more than the first-pass noise; the oracle check (true trajectory instead of first pass) would confirm.
- **Settings carry over to new words:** tuning → dev drops are small and similar (pretrained −0.009, fine-tuned −0.016), so neither is badly overfit. Fine-tuned vectors chose a stricter cutoff (0.80) and stronger smoothing (σ = 2).
- Fine-tuned results are on the same dev words and cases as pretrained, so the rows are directly comparable.
- **Test confirms dev:** fine-tuning +0.048 F1 (0.536 → 0.584, precision +0.048); temporal +0.002 on both vector sets, again mostly recall. The same small temporal gain on dev and test is a mild sign it is real, but it is still untested for significance. Dev and test agree closely (fine-tuned temporal 0.578 vs. 0.586), so no version is overfit to its tuning words.

Not implemented yet: the **definition variant** (LLM-written definition instead of averaged examples). Model choice and how to turn a definition into a vector are still open.

**Test words:** `run_subtask2` reports dev only. Test runs only with `final_test=True`, once settings are frozen (as in Split: test = final number only).

**Noticed, not changed:**
- Smoothing can't tell a model error from a real sudden change. A sense that truly appears in 1930 gets some of its share spread back into 1910, lowering the cutoff where the answer should be NO. σ is tuned on train, and the stable vs. newly appearing breakdown (decision 9) should show this.
- The trajectory is built from the same unlabeled sentences being classified (no labels used). This is allowed at test time but is worth saying in the report.

## Subtask 2 baseline (`src/baseline.py`)

Simplest working version, no year information. This is the number our temporal method has to beat.

- **train:** for each (word, sense) in the train words, average k example vectors → reference; score the word's other sentences by cosine similarity; pick the one cutoff with the best F1. Saved to `models/baseline.json`.
- **evaluate:** same on dev / test words with the learned cutoff. Truth labels come from `sense_id`: YES if the sentence's sense is the target sense, NO otherwise.
- **predict:** given example sentences of a sense + new sentences (text, start, end), encode with XL-LEXEME and answer yes/no.
- k = 3 (kept for the temporal comparison, see Subtask 2 decision 1); cutoffs tried 0.00–1.00 in steps of 0.01.
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

**Coverage:** every dev/test sentence takes part, and every word except 1 dev word (all its senses are too small to be a target). Senses with ≤ k sentences are never the target (128 of 1,239 dev senses, 123 of 1,187 test senses ≈ 10%). These are the rarest senses. Their sentences are still scored as NO cases for other senses. The k example sentences of each case are not scored.

**Noticed, not changed:**
-  most answers are NO, so accuracy looks better than the model is. Report F1 as the main metric.


## Fine-tuning XL-LEXEME (`src/embed.py`)

Second set of vectors from XL-LEXEME fine-tuned on our train words. Code written, **not run yet**. The pretrained vectors stay the main comparison.

Run (repo root): `python -m src.embed` → `splits/finetune_split.csv`, `models/xl-lexeme-ft/`, `cache/embeddings_ft.npy` + `cache/embeddings_ft_sentences.parquet`. `main.py` still uses the pretrained vectors.

**Why fine-tune at all:** our vectors so far come from the pretrained model used as-is (encoding only, no training). The task page (changeiskey.org, SemEval-2027) neither requires nor forbids fine-tuning; the Codabench rules page could not be read (check in a browser). Adapting a pretrained model to the target data is standard; XL-LEXEME is itself a fine-tuned XLM-R. Better vectors → more accurate first pass → less noisy trajectory for the temporal method.

| Decision | Choice | Why |
|---|---|---|
| Year in the model? | **No** (text + word position only) | Only one fine-tuned model, so it must not use the year: then baseline vs temporal on the fine-tuned vectors still isolates the year's effect at the decision step. A with-year model is only meaningful next to a no-year control. |
| Where saved | **Separate files** (`models/xl-lexeme-ft/`, `cache/embeddings_ft*`) | Pretrained vectors are kept for comparison (~2 h to recreate). |
| Which words | **Train words only, 80% "finetune" / 20% "holdout"** (seed 42, `splits/finetune_split.csv`, 1,500 / 374 words) | dev/test stay unseen. Cutoffs (and σ / adjust) must be picked on the **holdout** words: on words the model was trained on, scores look better than they are, so a cutoff picked there would be wrong for new words. |
| Pairs | **50 pairs per word: 25 same sense, 25 different sense, both sentences of the same word** (random, seed 42). Words with one sense (or no sense with 2+ sentences) skipped. | Standard for this model family: XL-LEXEME was trained on WiC-style same/different-meaning pairs (Cassotti et al., 2023), and WiC is balanced 50/50 by design (Pilehvar & Camacho-Collados, 2019), so the model doesn't learn to default to one answer. Same-word negatives are the hard cases the baseline gets wrong. In-batch-negative losses (most common for general embeddings) would mostly use other words = easy negatives. |
| Loss | **Contrastive loss on cosine distance**, sentence-transformers `ContrastiveLoss`, margin 0.5 (default) | Same kind of loss XL-LEXEME was trained with; binary labels (cosine-regression losses are for graded scores). **Check:** exact loss settings / margin of the original paper (taken from memory). |
| Training | **All weights, 1 epoch, lr 1e-5, batch 16 pairs, 10% warm-up, mixed precision** | One pass + small learning rate limit overfitting to the synthetic data. Mixed precision so the ~560M-parameter model fits on a 16 GB GPU (AMD RX 9070 XT). |

**Tuning with fine-tuned vectors:** `baseline.tuning_words(split, finetuned)` decides which words settings are picked on: all train words for the pretrained vectors (unchanged), only the 374 holdout words for the fine-tuned vectors. Used by `baseline.train` (cutoff) and `subtask2.run_subtask2` (σ, adjust). Fine-tuned results are saved separately (`models/baseline_ft.json`, `models/temporal_ft.json`), so the pretrained ones (0.74 etc.) are not overwritten. Run: `python -m src.embed`, then `python -m src.subtask2 --finetuned`.

**Noticed, not changed:**
- Synthetic data: the model may partly learn how the generator writes, not only real sense differences. Dev/test words are unseen, which limits but doesn't rule this out.
- The official Subtask 2 gives a **sense definition** as input; we use k example sentences instead because our dataset has no definitions. Our numbers are not directly comparable to official results.

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
- Sentence counts are a log-compressed version of the true frequency (see Data). Does smoothing still help when big real changes look small in the data?
- Do the 29 sentence texts with two different labels (e.g. "Bots.") noticeably lower the score?
- Is one random pick of example sentences stable, or do scores change with a different pick?
