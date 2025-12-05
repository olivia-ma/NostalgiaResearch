import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer

df = pd.read_csv('labeled_data.csv')

# ===============================================================
# 1. FIX RAW VALUES BEFORE FEATURE ENGINEERING
# ===============================================================

# --- Fix negative playtime ---
df['playtime_at_review'] = df['playtime_at_review'].clip(lower=0

# --- Fix negative review length (shouldn’t happen, but safe) ---
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

# 4. Game age bins
df['game_age_bin'] = pd.cut(
    df['game_age_at_review'],
    bins=[0, 5, 10, 100],
    labels=['recent', 'old', 'very_old'],
    include_lowest=True
)

# 5. Sentiment proxy
df['sentiment'] = df['recommended'].astype(int)


# ===============================================================
# 3. TEXT FEATURES (TF-IDF)
# ===============================================================

tfidf = TfidfVectorizer(max_features=100, stop_words='english', min_df=5)
tfidf_matrix = tfidf.fit_transform(df['review_text'].fillna(""))

tfidf_df = pd.DataFrame(
    tfidf_matrix.toarray(),
    columns=[f'tfidf_{w}' for w in tfidf.get_feature_names_out()]
)

df_features = pd.concat([df.reset_index(drop=True), tfidf_df], axis=1)


# ===============================================================
# 4. FINAL FEATURE MATRIX
# ===============================================================

feature_columns = [
    'playtime_at_review', 'playtime_ratio', 'low_engagement', 'high_engagement',
    'game_age_at_review', 'review_length', 'sentiment',
    'has_nostalgia_keywords', 'has_collectible_keywords'
] + [c for c in df_features.columns if c.startswith("tfidf_")]

X = df_features[feature_columns].copy()

# Normalize numerical features (safely)
scaler = StandardScaler()
numerical_cols = ['playtime_at_review', 'playtime_ratio', 'game_age_at_review', 'review_length']
X[numerical_cols] = scaler.fit_transform(X[numerical_cols])

# Save
X.to_csv('features.csv', index=False)
df_features['label'].to_csv('labels.csv', index=False)

print(f"Feature matrix shape: {X.shape}")
print("Sample cleaned values:")
print(df[['playtime_at_review', 'review_length', 'game_age_at_review']].head())
