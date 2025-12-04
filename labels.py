import pandas as pd
import numpy as np
import random
import re
from tqdm import tqdm

tqdm.pandas()

# ------------------------------
# 1. LOAD DATA
# ------------------------------

df = pd.read_csv('merged_data.csv')

# Convert UNIX timestamp → datetime
df["review_date"] = pd.to_datetime(df["timestamp"], unit='s')

# Extract review year from date instead of using df["year"]
df["review_year"] = df["review_date"].dt.year


# ------------------------------
# 2. KEYWORD DICTIONARIES
# ------------------------------

nostalgia_keywords = [
    "childhood", "nostalgia", "nostalgic", "growing up", "when i was a kid",
    "memory", "memories", "reminds me", "back in the day", "old times",
    "classic", "retro", "used to play", "played this as a kid", 
    "good old days", "grew up", "miss this", "brings back", 
    "loved this as a kid", "my childhood", "back when", "brings me back",
    "grew up playing", "from years ago", "remember playing", "revisiting",
    "revisit", "long time ago", "finally replaying", "my old favorite",
    "childhood favorite", "from my childhood"
]

collectible_keywords = [
    "collectible", "collecting", "collector", "rare", "rare card",
    "limited edition", "value increase", "worth money", "investment",
    "mint condition", "resale", "sealed", "graded", "psa", "collection",
    "complete", "completing", "own", "library", "catalog", "franchise",
    "completionist", "never played", "haven't played",
    "physical copy", "disc copy", "box copy", "cartridge", "manual",
    "vintage", "retro game market", "resale value", "value went up",
    "worth more now", "investment piece", "collector's item", "sealed copy",
    "grading", "BGS", "CGC", "slabbed", "variant cover"
]

# Compile regex for fast matching
nostalgia_regex = re.compile("|".join([re.escape(k) for k in nostalgia_keywords]), re.IGNORECASE)
collectible_regex = re.compile("|".join([re.escape(k) for k in collectible_keywords]), re.IGNORECASE)

price_terms = ["price", "worth", "value", "sell", "resell"]


# -------------------------------------------------------
# 3. AUTOMATED LABELING FUNCTION (FOUR CLASSES)
# -------------------------------------------------------

def detect_keywords(text):
    """Returns (has_nostalgia_keywords, has_collectible_keywords)."""
    if not isinstance(text, str):
        return False, False
    return bool(nostalgia_regex.search(text)), bool(collectible_regex.search(text))

def label_review(text):
    if not isinstance(text, str) or text.strip() == "":
        return "neither"

    lower = text.lower()

    nostalgia_match = bool(nostalgia_regex.search(text))
    collectible_match = bool(collectible_regex.search(text))

    # Price/value heuristic pushes toward collectible only if nostalgia not present
    if not collectible_match and not nostalgia_match:
        if any(term in lower for term in price_terms):
            collectible_match = True

    # Determine final class
    if nostalgia_match and collectible_match:
        return "both"
    if nostalgia_match:
        return "nostalgic"
    if collectible_match:
        return "collectible"
    
    return "neither"


# -------------------------------------------------------
# 4. APPLY LABELING + NEW FEATURE COLUMNS
# -------------------------------------------------------

df["label"] = df["review_text"].progress_apply(label_review)

# Boolean flags
df["has_nostalgia_keywords"] = df["review_text"].apply(lambda x: bool(nostalgia_regex.search(str(x))))
df["has_collectible_keywords"] = df["review_text"].apply(lambda x: bool(collectible_regex.search(str(x))))

# Game age at the time of review
# Uses 'year' (review year) and 'release_year' from your dataset
df["game_age_at_review"] = (df["review_year"] - df["release_year"]).clip(lower=0)

# Print counts
print(df['label'].value_counts())

# Save final dataset with the required columns included
df.to_csv('labeled_data.csv', index=False)


# -------------------------------------------------------
# 5. RANDOM MANUAL VERIFICATION (N = 100)
# -------------------------------------------------------

def print_manual_samples(df, n=100):
    labels = ["nostalgic", "collectible", "both", "neither"]
    samples = []

    per_label = n // len(labels)

    for label in labels:
        subset = df[df["label"] == label]
        if len(subset) > 0:
            samples.append(subset.sample(min(per_label, len(subset)), random_state=42))

    samples = pd.concat(samples).sample(frac=1, random_state=42)

    print("\n=== RANDOM MANUAL VERIFICATION SAMPLES ===\n")
    for _, row in samples.iterrows():
        print(f"[Label: {row['label']}]  {row['review_text'][:250]}")
        print("---")

# print_manual_samples(df, n=100)


# -------------------------------------------------------
# 6. SUMMARY STATISTICS
# -------------------------------------------------------

print("\n=== LABEL DISTRIBUTION ===\n")

label_counts = df["label"].value_counts()
total = len(df)

for label in ["nostalgic", "collectible", "both", "neither"]:
    count = label_counts.get(label, 0)
    pct = (count / total) * 100
    print(f"{label:12}: {count:6}  ({pct:5.2f}%)")

print("\nTotal reviews:", total)

summary = pd.DataFrame({
    "review_year": df["review_year"].describe(),
    "release_year": df["release_year"].describe(),
    "game_age_at_review": df["game_age_at_review"].describe()
})

print(summary)

