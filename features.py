import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer

df = pd.read_csv('labeled_data.csv')

# === BEHAVIORAL FEATURES ===

# 1. Playtime ratio (playtime at review / average playtime for that game)
df['playtime_ratio'] = df['playtime_at_review'] / (df['avg_playtime_hours'] + 1)

# 2. Low engagement flag (< 2 hours played)
df['low_engagement'] = (df['playtime_at_review'] < 2).astype(int)

# 3. High engagement flag (> 20 hours played)
df['high_engagement'] = (df['playtime_at_review'] > 20).astype(int)

# 4. Game age bin (old vs very old)
df['game_age_bin'] = pd.cut(df['game_age_at_review'], 
                             bins=[0, 5, 10, 100], 
                             labels=['recent', 'old', 'very_old'])

# 5. Review length (number of words)
df['review_length'] = df['review_text'].str.split().str.len()

# 6. Sentiment proxy (recommended = positive)
df['sentiment'] = df['recommended'].astype(int)

# === TEXT FEATURES ===

# Extract TF-IDF features from review text (top 100 words)
tfidf = TfidfVectorizer(max_features=100, stop_words='english', min_df=5)
tfidf_matrix = tfidf.fit_transform(df['review_text'])
tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), 
                        columns=[f'tfidf_{word}' for word in tfidf.get_feature_names_out()])

# Combine
df_features = pd.concat([df.reset_index(drop=True), tfidf_df], axis=1)

# === PREPARE FEATURE MATRIX ===

feature_columns = [
    'playtime_at_review', 'playtime_ratio', 'low_engagement', 'high_engagement',
    'game_age_at_review', 'review_length', 'sentiment',
    'has_nostalgia_keywords', 'has_collectible_keywords'
] + [col for col in df_features.columns if col.startswith('tfidf_')]

X = df_features[feature_columns]

# Normalize numerical features
scaler = StandardScaler()
numerical_cols = ['playtime_at_review', 'playtime_ratio', 'game_age_at_review', 'review_length']
X[numerical_cols] = scaler.fit_transform(X[numerical_cols])

# Save
X.to_csv('features.csv', index=False)
df_features['label'].to_csv('labels.csv', index=False)

print(f"Feature matrix shape: {X.shape}")
print(f"Features: {feature_columns[:10]}...")  # Show first 10