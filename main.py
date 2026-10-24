# Runs the whole pipeline. Run from the repo root: python main.py
from src.load_data import load_sentences
from src.clean_data import clean_sentences

# Step 1: load the data
df = load_sentences()
print(len(df), "sentences")
print(df.head())

# Step 2: clean
df = clean_sentences(df)
