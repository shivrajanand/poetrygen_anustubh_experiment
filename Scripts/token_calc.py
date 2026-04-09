from transformers import AutoTokenizer
import pandas as pd
import numpy as np

# =========================
# CONFIG
# =========================
MODEL_NAME = "unsloth/Qwen3.5-9B"
CSV_PATH = "Dataset/anustubh_hn_sa_train.csv"
COLUMNS = ['hi','sa']
USE_INSTRUCTION = True

ANUSHTUP_INSTRUCTION = """The goal is to generate Sanskrit verse that follows the anushtup meter rules for the given input text.
RULES:
Verse Rules:
The verse contains 32 syllables/akshara and 4 padas in total.
The verse is divided into 2 lines, each containing 16 syllables.
Each line is divided into 2 padas (quartets), each containing exactly 8 syllables.
The fifth syllable of every pada must be LAGHU or short.
The sixth syllable of every pada must be GURU or long.
The seventh syllable of the second and fourth pada must be HRASVA.
The seventh syllable of the first and third pada must be DEERGHA.

Syllable Rules:
LAGHU vowels: अ, इ, उ, ऋ, ऌ
GURU vowels: आ, ई, ऊ, ॠ, ॡ, ए, ऐ, ओ, औ
HRASVA vowels: अ, इ, उ, ऋ, ऌ
DEERGHA vowels: आ, ई, ऊ, ॠ, ॡ, ए, ऐ, ओ, औ

Syllable classification rules:
- A syllable is marked Laghu/Guru and Hrasva/Deergha based on the vowel it contains.
- Any syllable containing anusvāra (ं) or visarga (ः) is always Guru.
- Any syllable followed by a conjunct consonant (saṁyuktākṣara) is always Guru.
Now convert the given Hindi text into a Sanskrit Anushtup verse in Devanagari:"""


# =========================
# LOAD
# =========================
df = pd.read_csv(CSV_PATH)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# =========================
# TEXT BUILDER
# =========================
def build_text(row):
    parts = []
    
    if USE_INSTRUCTION:
        parts.append(ANUSHTUP_INSTRUCTION)
    
    for col in COLUMNS:
        val = str(row[col]) if pd.notna(row[col]) else ""
        parts.append(val)
    
    return "\n\n".join(parts)

# =========================
# TOKEN COUNT
# =========================
def compute_lengths(df, tokenizer):
    lengths = []
    
    for _, row in df.iterrows():
        text = build_text(row)
        tokens = tokenizer(text, add_special_tokens=False)["input_ids"]
        lengths.append(len(tokens))
    
    return np.array(lengths)

lengths = compute_lengths(df, tokenizer)

# =========================
# STATS
# =========================
def print_stats(lengths):
    print(f"\nModel: {MODEL_NAME}")
    print(f"Instruction used: {USE_INSTRUCTION}")
    print(f"Datset used: {CSV_PATH}")
    print(f"Columns Used: {COLUMNS}")
    
    print("Total samples:", len(lengths))
    print("Min:", lengths.min())
    print("Max:", lengths.max())
    print("Mean:", lengths.mean())
    print("Median:", np.median(lengths))
    
    print("\nPercentiles:")
    for p in [50, 75, 90, 95, 99]:
        print(f"{p}th:", np.percentile(lengths, p))

print_stats(lengths)