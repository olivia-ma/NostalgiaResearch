"""
merge_and_initial_analysis.py
Stage 2: Merge & Initial Analysis for Nostalgia vs Collectible project.

Outputs:
 - merged_data.csv       : cleaned merged review-level dataset (filtered)
 - game_stats.csv        : per-game aggregated statistics
 - reviews_non_english.csv (optional) : extracted non-English reviews if language filtering enabled
 - checkpoints: saved during processing

Notes:
 - Requires: pandas, numpy, regex, langdetect (optional but recommended), ftfy (optional)
 - If langdetect is not available, the code will continue and only flag language detection as unavailable.
"""

import re
import os
import sys
import math
import json
from collections import Counter

import pandas as pd
import numpy as np

# Optional packages
try:
    from langdetect import detect, DetectorFactory
    DetectorFactory.seed = 0
    LANGDETECT_AVAILABLE = True
except Exception:
    LANGDETECT_AVAILABLE = False

try:
    import ftfy
    FTFT_AVAILABLE = True
except Exception:
    FTFT_AVAILABLE = False

# ---------------------------
# Config
# ---------------------------
input_path ='files/'
GAMES_CSV = f'{input_path}games_master.csv'
REVIEWS_CSV = f'{input_path}steam_reviews_raw.csv'
OUT_MERGED = f'{input_path}merged_data.csv'
OUT_GAME_STATS = f'{input_path}game_stats.csv'
OUT_NON_ENG = f'{input_path}reviews_non_english.csv'

MIN_WORDS = 10               # remove reviews with fewer than this many words
MIN_REVIEWS_PER_GAME = 5     # drop games with fewer than this many reviews
SAVE_CHECKPOINT_EVERY = 10000  # rows

# Generational phrases to flag (expand if needed)
GENERATION_PHRASES = [
    r'back in the day', r'when i was', r'used to play', r'grew up', r'childhood',
    r'boomer', r'gen\s?x', r'gen\s?y', r'gen\s?z', r'millennial', r'nostalgia', r'nostalgic'
]

# URL / Steam profile pattern
URL_RE = re.compile(r'https?://\S+|www\.\S+')
STEAM_PROFILE_RE = re.compile(r'steamcommunity\.com/\S+|steamcommunity\.com/id/\S+|steamcommunity\.com/profiles/\S+')

# BBCode/HTML-ish patterns (Steam reviews frequently use [h1], [b], <br>, etc.)
BBCODE_RE = re.compile(r'\[/?[^\]]+\]|<[^>]+>')

# Excess punctuation / repeated char cleanup
REPEATED_PUNC_RE = re.compile(r'([!?.,]){2,}')
REPEAT_CHAR_RE = re.compile(r'(.)\1{3,}')  # e.g., "looooool" -> collapse later

# Non-letter-heavy heuristics (used to drop obviously non-textual reviews)
NON_LETTER_RATIO_THRESHOLD = 0.5  # if more than 50% non-letter characters, consider noisy

# ---------------------------
# Utility functions
# ---------------------------

def safe_read_csv(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found")
    try:
        return pd.read_csv(path, low_memory=False)
    except Exception as e:
        # try with engine python fallback
        return pd.read_csv(path, engine='python', low_memory=False)

def normalize_text(text):
    """Normalize unicode, remove URLs, BBCode/HTML, and extraneous whitespace/punctuation."""
    if pd.isna(text):
        return ''
    if FTFT_AVAILABLE:
        text = ftfy.fix_text(text)
    # convert to str
    text = str(text)
    # remove URLs / Steam profile links but record their presence elsewhere
    text = URL_RE.sub(' ', text)
    text = STEAM_PROFILE_RE.sub(' ', text)
    # remove bbcode/html
    text = BBCODE_RE.sub(' ', text)
    # collapse repeated punctuation
    text = REPEATED_PUNC_RE.sub(r'\1', text)
    # collapse long repeated chars to a max of 3
    text = REPEAT_CHAR_RE.sub(r'\1\1\1', text)
    # remove excessive non-ascii weirdness
    text = re.sub(r'[^\S\r\n]+', ' ', text)  # normalize whitespace
    return text.strip()

def contains_url(text):
    if pd.isna(text):
        return False
    return bool(URL_RE.search(str(text)) or STEAM_PROFILE_RE.search(str(text)))

def has_generation_phrase(text):
    if pd.isna(text):
        return False
    t = str(text).lower()
    return any(re.search(pat, t) for pat in GENERATION_PHRASES)

def is_mostly_nonletters(text, threshold=NON_LETTER_RATIO_THRESHOLD):
    if not text:
        return True
    letters = sum(1 for ch in text if ch.isalpha())
    return (1 - letters / max(1, len(text))) >= threshold

def detect_lang_safe(text):
    if not LANGDETECT_AVAILABLE:
        return None
    try:
        lang = detect(text)
        return lang
    except Exception:
        return None

# ---------------------------
# Load datasets
# ---------------------------
print("Loading datasets...")
games = safe_read_csv(GAMES_CSV)
reviews = safe_read_csv(REVIEWS_CSV)

print(f"Games rows: {len(games):,}, Reviews rows: {len(reviews):,}")

# Normalize appid column types and names
def ensure_appid_col(df, prefer='appid'):
    # common alternate names: appid, app_id
    if 'appid' in df.columns:
        df['appid'] = df['appid'].astype(int)
        return df
    if 'app_id' in df.columns:
        df = df.rename(columns={'app_id': 'appid'})
        df['appid'] = df['appid'].astype(int)
        return df
    # try to infer numeric first column
    for col in df.columns:
        if df[col].dtype.kind in 'iuf':
            df = df.rename(columns={col: 'appid'})
            df['appid'] = df['appid'].astype(int)
            return df
    raise KeyError("No appid/app_id column found in dataframe")

games = ensure_appid_col(games)
reviews = ensure_appid_col(reviews)

# Some reviews files use playtime_forever rather than playtime_hours
if 'playtime_forever' in reviews.columns and 'playtime_hours' not in reviews.columns:
    # sample suggests playtime in hours or fraction of hours; if it's minutes-units adjust?
    reviews = reviews.rename(columns={'playtime_forever': 'playtime_hours'})
# if there's playtime columns with /60 in raw sample, check and normalize: but assume already hours
# ensure numeric
if 'playtime_hours' in reviews.columns:
    reviews['playtime_at_review'] = pd.to_numeric(reviews.get('playtime_at_review', reviews['playtime_hours']), errors='coerce')
else:
    # fallback: if playtime_at_review exists use it; else create NaNs
    reviews['playtime_at_review'] = pd.to_numeric(reviews.get('playtime_at_review', np.nan), errors='coerce')

# timestamp normalization: ensure numeric epoch seconds
if 'timestamp' in reviews.columns:
    reviews['timestamp'] = pd.to_numeric(reviews['timestamp'], errors='coerce')
else:
    # try other names
    if 'timestamp_created' in reviews.columns:
        reviews['timestamp'] = pd.to_numeric(reviews['timestamp_created'], errors='coerce')

# ---------------------------
# Pre-cleaning: normalize game release_date to datetime & compute release_year
# ---------------------------
# Many release_date values are strings like "Oct 21, 2008"; attempt parse
games['release_date_parsed'] = pd.to_datetime(games.get('release_date', pd.NaT), errors='coerce')
# if year column exists, fallback
if 'year' in games.columns and games['release_date_parsed'].isna().any():
    try:
        games.loc[games['release_date_parsed'].isna(), 'release_date_parsed'] = pd.to_datetime(
            games.loc[games['release_date_parsed'].isna(), 'year'], format='%Y', errors='coerce'
        )
    except Exception:
        pass
games['release_year'] = games['release_date_parsed'].dt.year

# ---------------------------
# Merge
# ---------------------------
print("Merging datasets on appid (inner join)...")
df = reviews.merge(games, on='appid', how='inner', suffixes=('_rev', '_game'))
print(f"Merged rows: {len(df):,} (reviews matched to games)")

# ---------------------------
# Text cleaning & flags
# ---------------------------
print("Cleaning review text and creating flags...")

# unify review text column name: try several
if 'review_text' not in df.columns:
    # try common alternates
    alt = None
    for cand in ['review', 'text', 'content']:
        if cand in df.columns:
            alt = cand
            break
    if alt:
        df = df.rename(columns={alt: 'review_text'})
    else:
        raise KeyError("No review text column found in merged dataframe")

# ensure string type
df['review_text'] = df['review_text'].astype(str)

# Create flags before normalization
df['has_url'] = df['review_text'].apply(contains_url)
df['has_bbcode_html'] = df['review_text'].str.contains(r'\[/?[a-zA-Z0-9]+\]|<[^>]+>', regex=True, na=False)
df['contains_generational_phrase'] = df['review_text'].apply(has_generation_phrase)

# Normalize text
df['review_text_clean'] = df['review_text'].apply(normalize_text)

# Remove empty after cleaning
df['review_text_clean'] = df['review_text_clean'].fillna('').astype(str)

# detect language if possible
if LANGDETECT_AVAILABLE:
    print("langdetect available -> detecting language for each review (this may take a while)...")
    df['lang'] = df['review_text_clean'].apply(lambda t: detect_lang_safe(t) if len(t.split()) >= 3 else None)
else:
    print("langdetect not installed; skipping language detection. To enable, `pip install langdetect`.")
    df['lang'] = None

# flag mostly-nonletter (noisy short strings with emoji/garbage)
df['mostly_nonletters'] = df['review_text_clean'].apply(is_mostly_nonletters)

# ---------------------------
# Word count filter & duplicates
# ---------------------------
print("Filtering reviews with < %d words and dropping duplicates..." % MIN_WORDS)
df['word_count'] = df['review_text_clean'].str.split().str.len().fillna(0).astype(int)

# remove reviews with too few words
initial_count = len(df)
df = df[df['word_count'] >= MIN_WORDS]
filtered_count = len(df)
print(f"Removed {initial_count - filtered_count:,} short reviews (<{MIN_WORDS} words)")

# drop duplicate reviews (appid + text)
before_dupes = len(df)
df = df.drop_duplicates(subset=['appid', 'review_text_clean'])
after_dupes = len(df)
print(f"Dropped {before_dupes - after_dupes:,} duplicate reviews (appid + text)")

# remove extremely noisy reviews
before_noise = len(df)
df = df[~df['mostly_nonletters']]
after_noise = len(df)
print(f"Dropped {before_noise - after_noise:,} reviews flagged as mostly non-letter/noise")

# ---------------------------
# Option: filter to English-only (recommended if labels are English)
# ---------------------------
# Strategy: create two outputs. If >80% are English, we keep only English for modeling; otherwise keep all but flag.
eng_fraction = None
if LANGDETECT_AVAILABLE:
    lang_counts = df['lang'].value_counts(dropna=True)
    eng_fraction = (lang_counts.get('en', 0) / max(1, lang_counts.sum()))
    print("Language distribution (top):")
    print(lang_counts.head(10))
    print(f"English fraction among detected: {eng_fraction:.3f}")

    # Decision heuristic: if detected english fraction >= 0.7, keep only English for modeling (you can change)
    if eng_fraction >= 0.7:
        print("Filtering to English-only reviews (lang == 'en') because English fraction >= 0.70")
        non_english = df[df['lang'] != 'en']
        if len(non_english):
            non_english.to_csv(OUT_NON_ENG, index=False)
            print(f"Saved {len(non_english):,} non-English reviews to {OUT_NON_ENG}")
        df = df[df['lang'] == 'en']
    else:
        print("English fraction < 0.7 -> keeping non-English reviews but they remain flagged 'lang'.")
else:
    print("langdetect not available -> not filtering by language.")

# ---------------------------
# Remove games with too few reviews
# ---------------------------
print("Aggregating per-game review counts and filtering games with fewer than %d reviews..." % MIN_REVIEWS_PER_GAME)
game_counts = df.groupby('appid').size().rename('review_count').reset_index()
game_counts['review_count'] = game_counts['review_count'].astype(int)

# merge counts back
df = df.merge(game_counts, on='appid', how='left')

# keep games with >= threshold
before_game_prune = len(df)
valid_games = game_counts[game_counts['review_count'] >= MIN_REVIEWS_PER_GAME]['appid'].unique()
df = df[df['appid'].isin(valid_games)]
after_game_prune = len(df)
print(f"Removed {before_game_prune - after_game_prune:,} reviews from games with <{MIN_REVIEWS_PER_GAME} reviews")

# ---------------------------
# Compute game-level stats
# ---------------------------
print("Computing game-level stats (avg playtime, recommended rate, review counts, release year counts)...")
# ensure recommended is boolean/numeric
if 'recommended' in df.columns:
    # recommended may be boolean True/False or text; normalize
    df['recommended_flag'] = df['recommended'].apply(lambda x: 1 if str(x).lower() in ['true', '1', 't', 'yes'] else (0 if str(x).lower() in ['false', '0', 'f', 'no'] else np.nan))
else:
    df['recommended_flag'] = np.nan

# use playtime_at_review as main measure
df['playtime_at_review'] = pd.to_numeric(df['playtime_at_review'], errors='coerce')

game_stats = df.groupby('appid').agg(
    review_count=('review_text_clean', 'count'),
    avg_playtime_hours=('playtime_at_review', 'mean'),
    median_playtime_hours=('playtime_at_review', 'median'),
    recommended_rate=('recommended_flag', 'mean'),
    release_year=('release_year', 'first'),
    game_name=('name', 'first')
).reset_index()

# Save game-level stats
game_stats.to_csv(OUT_GAME_STATS, index=False)
print(f"Saved game stats to {OUT_GAME_STATS} ({len(game_stats):,} games)")

# ---------------------------
# Merge back game-level stats into df (for modeling convenience)
# ---------------------------
df = df.merge(game_stats[['appid', 'avg_playtime_hours', 'review_count', 'recommended_rate']], on='appid', how='left')

# ---------------------------
# Save final merged dataset
# ---------------------------
print(f"Saving merged dataset to {OUT_MERGED} (rows: {len(df):,})")
df.to_csv(OUT_MERGED, index=False)

# Print quick summary
n_games = df['appid'].nunique()
n_reviews = len(df)
print("Final dataset summary:")
print(f" - Reviews: {n_reviews:,}")
print(f" - Unique games: {n_games:,}")
print("Top 10 games by review count:")
print(game_stats.sort_values('review_count', ascending=False).head(10).to_string(index=False))

# Quick stats: reviews per release year
if 'release_year' in df.columns:
    year_counts = df['release_year'].value_counts().sort_index()
    print("\nReviews per release year (sample):")
    print(year_counts.tail(10).to_string())

print("\nDone.")
