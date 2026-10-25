# Fix or drop rows where start/end don't point at the target word.
import re


def span_looks_ok(span, word):
    # same first 3 letters, so inflected forms like "ship" -> "shipped" count as ok.
    # hyphens and spaces are ignored so "re-starts" matches "restart"
    span = span.lower().replace("-", "").replace(" ", "")
    word = word.lower().replace("-", "").replace(" ", "")
    n = min(3, len(word))
    return len(span) > 0 and span[:n] == word[:n]


def find_word(text, word):
    # the word plus any ending ("predicate", "predicated"), as a whole word, any case
    match = re.search(r"\b" + re.escape(word) + r"\w*", text, flags=re.IGNORECASE)
    if match:
        return match.start(), match.end()
    return None


def clean_sentences(df):
    df = df.copy()
    fixed, dropped = 0, []

    for i, row in df.iterrows():
        span = row["text"][row["start"]:row["end"]]
        if span_looks_ok(span, row["lemma"]):
            continue

        position = find_word(row["text"], row["lemma"])
        if position:
            df.at[i, "start"], df.at[i, "end"] = position
            fixed += 1
        else:
            dropped.append(i)

    df = df.drop(index=dropped).reset_index(drop=True)
    print(f"fixed {fixed} rows, dropped {len(dropped)} rows, kept {len(df)} rows")
    return df