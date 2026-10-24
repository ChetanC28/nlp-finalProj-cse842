# Look for problems in the dataset before cleaning.
# Run from the repo root (after main.py has made the cache): python src/explore_data.py
import pandas as pd

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", None)
pd.set_option("display.max_colwidth", 80)

df = pd.read_parquet("cache/sentences.parquet")

print("=== 1. Random rows ===")
print(df.sample(10, random_state=0))

print("\n=== 2. Missing values ===")
print(df.isna().sum())

print("\n=== 3. Duplicates ===")
print("duplicate sentences for the same word:", df.duplicated(subset=["lemma", "text"]).sum())
print("fully identical rows:", df.duplicated().sum())

print("\n=== 4. Value ranges ===")
print(df["year"].value_counts().sort_index())
print(df["pos"].value_counts())

print("\n=== 5. Is text[start:end] the target word? ===")
df["span"] = df.apply(lambda r: r["text"][r["start"]:r["end"]], axis=1)
bad = df[df["span"].str.lower() != df["lemma"].str.lower()]
empty = bad[bad["span"].str.strip() == ""]
print("span does not match the word:", len(bad))
print("span is empty:", len(empty))
print(bad[["lemma", "span", "text"]].sample(15, random_state=0))
