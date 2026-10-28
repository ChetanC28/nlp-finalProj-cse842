# Subtask 1: Sense assignment and semantic change ranking

## Setup

We evaluate XL-LEXEME occurrence embeddings using unsupervised average-linkage agglomerative clustering with cosine distance. Clusters represent inferred senses for Subtask 1a. For Subtask 1b, we measure change by summing Jensen–Shannon divergences between consecutive **observed** time-period sense distributions and evaluate word-level rankings using Spearman correlation.

The experiments use the synthetic, single-label sense annotations distributed with the project data. These scores are **not** official SemEval leaderboard results. Gold sense labels are used for scoring and training-set hyperparameter selection, not for building clusters or estimating temporal priors.

An existing word-level train/development/test split is maintained. One training word, `jock`, is unavailable in the cached embeddings and is excluded by split reconciliation; no word is reassigned to a different partition. The test partition was not evaluated in these runs.

## Experiment design

- **Original baseline:** threshold 0.35, no temporal adjustment.
- **Threshold tuning:** average-linkage cosine thresholds 0.25–0.65 in steps of 0.05, using 500 fixed training words in the extended experiment.
- **Temporal adjustment:** initial cluster centroids and predicted sense frequencies are combined with local time-window priors; an occurrence is excluded from its own local prior.
- **Confidence-gated adjustment:** reassignments are restricted to occurrences with low similarity-margin confidence.
- **Selection:** configurations for 1a and 1b are selected independently using only the training words.
- **Validation:** 234 development words; paired bootstrap comparisons use 1,000 resamples. Calendar-adjacent-only JSD is included as a diagnostic, not substituted for the observed-period baseline.

## Main results

| Configuration | Development BCubed F1 (1a) | Development Spearman ρ (1b) |
| --- | ---: | ---: |
| Original static clustering, threshold 0.35 | 0.6271 | 0.4000 |
| Tuned static clustering, threshold 0.45 | **0.6335** | **0.4063** |
| Original temporal reassignment, threshold 0.45, strength 0.20 | 0.6293 | 0.3975 |

The train-selected configuration from the extended experiment is **static clustering at threshold 0.45 for both tasks**. The wider search showed lower training scores at thresholds 0.50–0.65. Confidence-gated temporal variants did not exceed the static training baseline.

Paired development bootstrap comparisons of threshold 0.45 versus 0.35 show ΔF1 = +0.00637 (95% interval −0.00377 to +0.01665) and Δρ = +0.00636 (95% interval −0.06238 to +0.07716). Both intervals include zero; the measured gains are modest and not statistically conclusive.

## Temporal error analysis

On 500 training words, the confidence-gated configuration with temporal strength 0.35 and gate 0.025 improved BCubed F1 on 121 words, lowered it on 239 words, and left it unchanged on 140 words. Larger reassignment counts were associated with larger mean F1 decreases. Words with missing periods did not have larger mean decreases than those with full coverage. These associations are diagnostic, not causal claims.

The temporal experiments are retained as ablations and negative results. They do not establish an improvement over the tuned static model.

## Reproduction

Place `embeddings.npy` and `embeddings_sentences.parquet` in the parent directory of the repository, or change `--data-dir` to the location of the cached embeddings. From the repository root:

```bash
python scripts/subtask1_round2.py \
  --repo-root . \
  --data-dir .. \
  --output-dir results/subtask1_round2 \
  --train-words 500 \
  --include-reference-temporal
```

The script writes threshold sweeps, temporal sweeps, selected training parameters, development comparisons, per-word diagnostics, and a run manifest. Test evaluation requires the explicit `--final-test` flag and should only be run after freezing the configuration.

For the unit tests:

```bash
PYTHONPATH=scripts python tests/test_round2.py
PYTHONPATH=scripts python -m unittest discover -s tests -p 'test_subtask1_complete.py'
```
