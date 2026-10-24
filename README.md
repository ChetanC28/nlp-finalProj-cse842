# nlp-finalProj-cse842

Semantic change across multiple time periods (SemEval-2027 Task 3, Subtasks 1 and 2), using XL-LEXEME embeddings with temporal trajectory smoothing.

## Dataset

Synthetic English diachronic dataset of Saponara et al. (2026): https://doi.org/10.5281/zenodo.21703081

Extract it into `NLP Dataset/` (gitignored) so it contains `generated_sentences/` and `metadata/`.

## Setup

Requires Python 3.11 (`brew install python@3.11` on macOS).

```bash
./setup.sh                  # creates .venv and installs pinned versions
source .venv/bin/activate   # run this in every new terminal
```
