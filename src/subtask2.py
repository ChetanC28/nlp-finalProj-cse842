# Subtask 2: for one sense of a word, decide which sentences use that sense.
# Skeleton only. Each DECISION comment is a choice the team needs to make before we write the code.
#
# Assumes the encoding step gave us (same row order):
#   cache/embeddings.npy               one vector per sentence
#   cache/embeddings_sentences.parquet the cleaned sentences

PERIODS = [1810, 1830, 1850, 1870, 1890, 1910, 1930, 1950, 1970, 1990]


# --- Step 1: reference for a sense ---

def build_reference(vectors, sense_rows, k):
    """Pick k sentences of the sense and turn them into one reference vector.
    Returns the reference vector and which rows were used (so they are not scored).

    WHY: the model needs to know what the target sense "looks like". The dataset has no
         written definitions, only labeled sentences, so a few known examples describe it.
         Averaging cancels out the details of each single sentence and keeps what they share.
         The examples are not scored, because the model would get them right for free.

    DECISION 1: k, how many example sentences. Not fixed by the task.
                Fewer = more senses usable; more = more reliable reference.
    DECISION 2: where the k examples come from: random from any period, or earliest period only.
    DECISION 3: what to do if a sense has <= k sentences: skip the sense, or use what's there.
    (Averaging the k vectors into a prototype is standard: Cassotti & Tahmasebi 2025.)
    """


# --- Step 2: score sentences ---

def score_sentences(vectors, reference, rows):
    """Similarity of each sentence to the reference (cosine similarity, standard for XL-LEXEME).

    WHY: turns "does this sentence use the sense?" into a number. XL-LEXEME was trained so
         that same-meaning uses have high cosine similarity, so a high score = likely the sense.

    DECISION 4: which sentences get scored: all of the word's other sentences
                (other senses = the "no" cases), or something else.
    """


# --- Step 3a: baseline decision ---

def predict_baseline(scores, cutoff):
    """Yes if score >= cutoff. One cutoff for every period.

    WHY: the standard method that ignores the year. We need it to compare against:
         our method only counts as an improvement if it beats this.
    """


# --- Step 3b: temporal decision (our method) ---

def sense_trajectory(predictions, periods):
    """From a first pass of yes/no answers, how common the sense is in each of the 10 periods.

    WHY: this is the "sense trajectory" from the proposal (sec 4.2). We don't know the true
         labels at test time, so we estimate how common the sense is from the baseline's answers.

    DECISION 5: measure it as share of the word's sentences in that period ("30% yes"),
                or raw count ("12 yes").
    """


def smooth(trajectory, strength):
    """Smooth the 10-point curve so single-period spikes count as noise.

    WHY: the proposal's main assumption: meanings rise and fall gradually over decades.
         A sudden jump in one period is more likely a model mistake than real change.

    DECISION 6: smoothing method: moving average over neighbouring periods, or another method.
                strength = how much smoothing (none ... a lot).
    """


def period_cutoffs(smoothed, base_cutoff, adjust):
    """Turn the smoothed curve into one cutoff per period.
    Sense common in a period -> lower cutoff there; rare -> higher.

    WHY: this is where the year actually changes the answer. A borderline sentence is more
         likely the sense in a period where the sense is common, and less likely where it is rare.

    DECISION 7: the formula linking "how common" to "how much the cutoff moves".
    """


def predict_temporal(scores, periods, base_cutoff, strength, adjust):
    """Our method: baseline first pass -> trajectory -> smooth -> per-period cutoffs -> yes/no.

    WHY: puts the steps above together so it can be compared directly with predict_baseline.
         Both use the same scores, so any difference comes only from the decision step.
    """


# --- Step 4: evaluation ---

def evaluate(results):
    """F1 for the yes/no answers against the true sense labels.

    WHY: F1 is what the proposal promises (sec 4.5). It balances finding the sense's
         sentences (recall) with not wrongly saying yes (precision).

    DECISION 8: how to average: per sense -> per word -> across words (proposal says
                "per word, averaged across words"), or another way.
    DECISION 9: also report separately for stable / newly appearing / rare senses
                (promised in the proposal). Need to define each group using metadata/.
    """


# --- Step 5: tuning and running ---

def tune(train_cases):
    """Pick the best settings using the train words only.

    WHY: cutoff and smoothing strength have no "correct" value; we have to try values and keep
         the best. Doing that on train words only keeps the dev/test scores honest.

    DECISION 10: which values to try for cutoff, strength, adjust.
    DECISION 11: tune the baseline cutoff first and then the temporal settings,
                 or tune everything together.
    """


def run_subtask2(df, vectors, split):
    """Whole Subtask 2: build references, score, tune on train, report dev and test.

    WHY: one entry point so main.py can run Subtask 2 with a single call.

    DECISION 12: picking references is random. Run once, or several times
                 with different random picks and average (more reliable).
    """
