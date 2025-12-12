import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer

df = pd.read_csv('files/labeled_data.csv')

# ===============================================================
# 1. FIX RAW VALUES BEFORE FEATURE ENGINEERING
# ===============================================================

# --- Fix negative playtime ---
df['playtime_at_review'] = df['playtime_at_review'].clip(lower=0)

# --- Fix negative review length (shouldn't happen, but safe) ---
df['review_length'] = df['review_text'].fillna("").str.split().str.len().clip(lower=0)

# --- Fix negative game age at review ---
df['game_age_at_review'] = (df['review_year'] - df['release_year']).clip(lower=0)


# ===============================================================
# 2. BEHAVIORAL FEATURES
# ===============================================================

# 1. Playtime ratio (avoid negative; avoid divide-by-zero)
df['playtime_ratio'] = df['playtime_at_review'] / (df['avg_playtime_hours'].clip(lower=1))

# 2. Low engagement flag (< 2 hours)
df['low_engagement'] = (df['playtime_at_review'] < 2).astype(int)

# 3. High engagement flag (> 20 hours)
df['high_engagement'] = (df['playtime_at_review'] > 20).astype(int)

# 4. Game age bins (encoded as dummies)
df['game_age_bin'] = pd.cut(
    df['game_age_at_review'],
    bins=[0, 5, 10, 100],
    labels=['recent', 'old', 'very_old'],
    include_lowest=True
)
age_dummies = pd.get_dummies(df['game_age_bin'], prefix='age')

# 5. Sentiment proxy
df['sentiment'] = df['recommended'].astype(int)

# 6. Review length features
df['very_short_review'] = (df['review_length'] < 20).astype(int)
df['very_long_review'] = (df['review_length'] > 200).astype(int)


# ===============================================================
# 3. TEXT FEATURES (TF-IDF) - EXCLUDING KEYWORD DICTIONARY WORDS
# ===============================================================

# CRITICAL: Remove nostalgia/collectible keywords from vocabulary
# to prevent data leakage
nostalgia_keywords = [
    "childhood", "nostalgia", "nostalgic", "growing up", "memory", 
    "memories", "reminds me", "back in the day", "old times",
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
    "grading", "bgs", "cgc", "slabbed", "variant cover"
]

# Combine all keywords to exclude
excluded_keywords = set(nostalgia_keywords + collectible_keywords)

# Custom stop words = english stopwords + our keyword dictionaries
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
custom_stop_words = list(ENGLISH_STOP_WORDS) + list(excluded_keywords)

tfidf = TfidfVectorizer(
    max_features=100, 
    stop_words=custom_stop_words,
    min_df=5,
    ngram_range=(1, 2)  # include bigrams for richer context
)
tfidf_matrix = tfidf.fit_transform(df['review_text'].fillna(""))

tfidf_df = pd.DataFrame(
    tfidf_matrix.toarray(),
    columns=[f'tfidf_{w}' for w in tfidf.get_feature_names_out()]
)


# ===============================================================
# 4. FINAL FEATURE MATRIX (NO KEYWORD FLAGS!)
# ===============================================================

# Combine all features
df_combined = pd.concat([
    df.reset_index(drop=True), 
    age_dummies.reset_index(drop=True),
    tfidf_df
], axis=1)

# Define feature columns - EXCLUDING has_nostalgia_keywords and has_collectible_keywords
feature_columns = [
    'playtime_at_review', 'playtime_ratio', 'low_engagement', 'high_engagement',
    'game_age_at_review', 'review_length', 'sentiment',
    'very_short_review', 'very_long_review'
] + list(age_dummies.columns) + [c for c in tfidf_df.columns]

X = df_combined[feature_columns].copy()

# Normalize numerical features
scaler = StandardScaler()
numerical_cols = ['playtime_at_review', 'playtime_ratio', 'game_age_at_review', 'review_length']
X[numerical_cols] = scaler.fit_transform(X[numerical_cols])

# Save
X.to_csv('files/features.csv', index=False)
df['label'].to_csv('files/labels.csv', index=False)

print(f"Feature matrix shape: {X.shape}")
print(f"Features used: {len(feature_columns)}")
print("\nSample cleaned values:")
print(df[['playtime_at_review', 'review_length', 'game_age_at_review']].head())
print("\n✓ NO KEYWORD FLAGS INCLUDED - preventing data leakage")