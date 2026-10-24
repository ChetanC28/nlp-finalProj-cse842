# Runs the whole pipeline. Run from the repo root: python main.py
from src.load_data import load_sentences

# Step 1: load the data
df = load_sentences()
print(len(df), "sentences")
print(df.head())

# Step 2: clean (todo)
