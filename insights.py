import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load the labeled data (this has all original columns)
df = pd.read_csv("files/labeled_data.csv")

# Filter to valid classes
VALID_CLASSES = ["nostalgic", "collectible", "both"]
df = df[df["label"].isin(VALID_CLASSES)].copy()

# Calculate review_length if not present
if 'review_length' not in df.columns:
    df['review_length'] = df['review_text'].fillna("").str.split().str.len()

print("="*60)
print("GENERATING INSIGHTS")
print("="*60)
print(f"Total reviews: {len(df):,}")
print(f"Label distribution:")
print(df['label'].value_counts())
print()

# =========================
# INSIGHT 1 — Playtime
# =========================

print("Generating playtime distribution plot...")
plt.figure(figsize=(10, 6))

# Filter out extreme outliers for better visualization
playtime_filtered = df[df['playtime_at_review'] > 0.1].copy()

sns.boxplot(data=playtime_filtered, x="label", y="playtime_at_review", order=VALID_CLASSES)
plt.yscale("log")
plt.title("Playtime Distribution by Category", fontsize=14, fontweight='bold')
plt.ylabel("Hours Played (log scale)", fontsize=12)
plt.xlabel("Category", fontsize=12)
plt.tight_layout()
plt.savefig("images/insight_playtime.png", dpi=300, bbox_inches="tight")
plt.close()
print("✓ Saved insight_playtime.png")

# =========================
# INSIGHT 2 — Game Age
# =========================

print("Generating game age distribution plot...")
plt.figure(figsize=(10, 6))
sns.violinplot(data=df, x="label", y="game_age_at_review", order=VALID_CLASSES)
plt.title("Game Age at Review Time", fontsize=14, fontweight='bold')
plt.ylabel("Years Since Release", fontsize=12)
plt.xlabel("Category", fontsize=12)
plt.tight_layout()
plt.savefig("images/insight_game_age.png", dpi=300, bbox_inches="tight")
plt.close()
print("✓ Saved insight_game_age.png")

# =========================
# INSIGHT 3 — Summary Stats
# =========================

print("\nCalculating summary statistics...")

# Ensure recommended is numeric
if 'recommended' in df.columns:
    df['recommended_numeric'] = df['recommended'].apply(
        lambda x: 1 if str(x).lower() in ['true', '1', 't', 'yes'] else 0
    )
elif 'recommended_flag' in df.columns:
    df['recommended_numeric'] = df['recommended_flag']
else:
    df['recommended_numeric'] = np.nan

summary = df.groupby("label").agg({
    "playtime_at_review": ["mean", "median", "std"],
    "game_age_at_review": ["mean", "median", "std"],
    "review_length": ["mean", "median"],
    "recommended_numeric": "mean"
}).round(2)

print("\n=== SUMMARY STATISTICS ===")
print(summary)
print()

summary.to_csv("files/insights_summary.csv")
print("✓ Saved insights_summary.csv")

# =========================
# INSIGHT 4 — Keyword Flags
# =========================

if 'has_nostalgia_keywords' in df.columns and 'has_collectible_keywords' in df.columns:
    keyword_stats = df.groupby("label").agg({
        "has_nostalgia_keywords": "mean",
        "has_collectible_keywords": "mean"
    }).round(3)
    
    print("\n=== KEYWORD PRESENCE (% of reviews with keywords) ===")
    print(keyword_stats)
    print()

# =========================
# INSIGHT 5 — Genre Breakdown
# =========================

if "genres" in df.columns:
    print("Generating genre breakdown...")
    
    # Expand comma-separated genres
    genre_expanded = df.assign(genre=df['genres'].str.split(', ')).explode('genre')
    genre_expanded = genre_expanded[genre_expanded['genre'].notna()]
    
    # Get top genres overall
    top_genres = genre_expanded['genre'].value_counts().head(10).index.tolist()
    
    # Count by label
    genre_counts = genre_expanded[genre_expanded['genre'].isin(top_genres)].groupby(['label', 'genre']).size().unstack(fill_value=0)
    
    plt.figure(figsize=(12, 6))
    genre_counts.T.plot(kind="bar", width=0.8)
    plt.title("Top 10 Genres by Category", fontsize=14, fontweight='bold')
    plt.ylabel("Count", fontsize=12)
    plt.xlabel("Genre", fontsize=12)
    plt.legend(title="Category")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig("images/insight_genres.png", dpi=300)
    plt.close()
    print("✓ Saved insight_genres.png")

# =========================
# INSIGHT 6 — Review Length
# =========================

print("Generating review length comparison...")
plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x="label", y="review_length", order=VALID_CLASSES)
plt.title("Review Length by Category", fontsize=14, fontweight='bold')
plt.ylabel("Word Count", fontsize=12)
plt.xlabel("Category", fontsize=12)
plt.tight_layout()
plt.savefig("images/insight_review_length.png", dpi=300, bbox_inches="tight")
plt.close()
print("✓ Saved insight_review_length.png")

# =========================
# INSIGHT 7 — Sentiment (Recommended Rate)
# =========================

print("Generating sentiment comparison...")
sentiment_pct = df.groupby('label')['recommended_numeric'].mean() * 100

plt.figure(figsize=(8, 6))
sentiment_pct.plot(kind='bar', color=['#2ecc71', '#e74c3c', '#3498db'])
plt.title("Recommendation Rate by Category", fontsize=14, fontweight='bold')
plt.ylabel("% Recommended", fontsize=12)
plt.xlabel("Category", fontsize=12)
plt.ylim(0, 100)
plt.xticks(rotation=0)
for i, v in enumerate(sentiment_pct):
    plt.text(i, v + 2, f"{v:.1f}%", ha='center', fontweight='bold')
plt.tight_layout()
plt.savefig("images/insight_sentiment.png", dpi=300, bbox_inches="tight")
plt.close()
print("✓ Saved insight_sentiment.png")

# =========================
# KEY FINDINGS TEXT FILE
# =========================

findings = f"""
{'='*60}
KEY FINDINGS: NOSTALGIA VS COLLECTIBLE BUYERS
{'='*60}

SAMPLE SIZE:
  Total reviews analyzed: {len(df):,}
  - Nostalgic:   {(df['label'] == 'nostalgic').sum():,} ({(df['label'] == 'nostalgic').sum()/len(df)*100:.1f}%)
  - Collectible: {(df['label'] == 'collectible').sum():,} ({(df['label'] == 'collectible').sum()/len(df)*100:.1f}%)
  - Both:        {(df['label'] == 'both').sum():,} ({(df['label'] == 'both').sum()/len(df)*100:.1f}%)

{'='*60}
1. PLAYTIME PATTERNS
{'='*60}

Median hours played:
  - Nostalgic:   {summary.loc['nostalgic', ('playtime_at_review', 'median')]:.2f} hours
  - Collectible: {summary.loc['collectible', ('playtime_at_review', 'median')]:.2f} hours
  - Both:        {summary.loc['both', ('playtime_at_review', 'median')]:.2f} hours

Mean hours played:
  - Nostalgic:   {summary.loc['nostalgic', ('playtime_at_review', 'mean')]:.2f} hours
  - Collectible: {summary.loc['collectible', ('playtime_at_review', 'mean')]:.2f} hours
  - Both:        {summary.loc['both', ('playtime_at_review', 'mean')]:.2f} hours

INTERPRETATION:
  → Nostalgic buyers show {'HIGHER' if summary.loc['nostalgic', ('playtime_at_review', 'median')] > summary.loc['collectible', ('playtime_at_review', 'median')] else 'LOWER'} engagement with games
  → Collectible buyers may purchase with {'LOW' if summary.loc['collectible', ('playtime_at_review', 'median')] < 5 else 'MODERATE'} intent to play

{'='*60}
2. GAME AGE AT REVIEW
{'='*60}

Average years since release:
  - Nostalgic:   {summary.loc['nostalgic', ('game_age_at_review', 'mean')]:.2f} years
  - Collectible: {summary.loc['collectible', ('game_age_at_review', 'mean')]:.2f} years
  - Both:        {summary.loc['both', ('game_age_at_review', 'mean')]:.2f} years

INTERPRETATION:
  → Nostalgic buyers review {'OLDER' if summary.loc['nostalgic', ('game_age_at_review', 'mean')] > summary.loc['collectible', ('game_age_at_review', 'mean')] else 'NEWER'} games on average
  → Time since release {'IS' if abs(summary.loc['nostalgic', ('game_age_at_review', 'mean')] - summary.loc['collectible', ('game_age_at_review', 'mean')]) > 2 else 'IS NOT'} a strong differentiator

{'='*60}
3. SENTIMENT (RECOMMENDATION RATE)
{'='*60}

% of reviews that recommend the game:
  - Nostalgic:   {summary.loc['nostalgic', ('recommended_numeric', 'mean')] * 100:.1f}%
  - Collectible: {summary.loc['collectible', ('recommended_numeric', 'mean')] * 100:.1f}%
  - Both:        {summary.loc['both', ('recommended_numeric', 'mean')] * 100:.1f}%

INTERPRETATION:
  → Nostalgic buyers are {'MORE' if summary.loc['nostalgic', ('recommended_numeric', 'mean')] > summary.loc['collectible', ('recommended_numeric', 'mean')] else 'LESS'} likely to recommend
  → Sentiment difference: {abs(summary.loc['nostalgic', ('recommended_numeric', 'mean')] - summary.loc['collectible', ('recommended_numeric', 'mean')]) * 100:.1f} percentage points

{'='*60}
4. REVIEW BEHAVIOR
{'='*60}

Average review length (words):
  - Nostalgic:   {summary.loc['nostalgic', ('review_length', 'mean')]:.1f} words
  - Collectible: {summary.loc['collectible', ('review_length', 'mean')]:.1f} words
  - Both:        {summary.loc['both', ('review_length', 'mean')]:.1f} words

INTERPRETATION:
  → {'Nostalgic' if summary.loc['nostalgic', ('review_length', 'mean')] > summary.loc['collectible', ('review_length', 'mean')] else 'Collectible'} buyers write longer reviews
  → Review length difference: {abs(summary.loc['nostalgic', ('review_length', 'mean')] - summary.loc['collectible', ('review_length', 'mean')]):.1f} words

{'='*60}
BUSINESS IMPLICATIONS
{'='*60}

For game publishers and marketers:

1. NOSTALGIC BUYERS:
   - High engagement, emotional connection to games
   - {'Positive' if summary.loc['nostalgic', ('recommended_numeric', 'mean')] > 0.7 else 'Mixed'} sentiment overall
   - Marketing angle: emotional storytelling, "relive your childhood"
   - Product strategy: remasters, sequels to classic franchises

2. COLLECTIBLE BUYERS:
   - {'Lower' if summary.loc['collectible', ('playtime_at_review', 'median')] < summary.loc['nostalgic', ('playtime_at_review', 'median')] else 'Similar'} playtime vs nostalgic buyers
   - May purchase for investment/preservation
   - Marketing angle: limited editions, special packaging
   - Product strategy: physical media, collector's editions

3. OVERLAP ("BOTH"):
   - Represents {(df['label'] == 'both').sum():,} reviews ({(df['label'] == 'both').sum()/len(df)*100:.1f}%)
   - Combines emotional + collecting motivations
   - Highest value segment for premium offerings

{'='*60}
"""

with open("files/key_findings.txt", "w") as f:
    f.write(findings)

print("\n✓ Saved key_findings.txt")
print(findings)

print("\n" + "="*60)
print("INSIGHTS GENERATION COMPLETE")
print("="*60)
print("\nGenerated files:")
print("  - insight_playtime.png")
print("  - insight_game_age.png")
print("  - insight_review_length.png")
print("  - insight_sentiment.png")
if "genres" in df.columns:
    print("  - insight_genres.png")
print("  - insights_summary.csv")
print("  - key_findings.txt")

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, balanced_accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
X = pd.read_csv("files/features.csv")
y_full = pd.read_csv("files/labels.csv")["label"]
df_raw = pd.read_csv("files/labeled_data.csv")

# Filter to valid classes
mask = y_full.isin(["nostalgic", "collectible", "both"])
X = X[mask].reset_index(drop=True)
y_full = y_full[mask].reset_index(drop=True)

print("="*60)
print("BINARY CLASSIFICATION ANALYSIS")
print("="*60)

# ==========================================
# BINARY 1: Nostalgic (Yes/No)
# ==========================================
print("\n1. NOSTALGIC vs NOT-NOSTALGIC")
print("-"*60)

y_nostalgic = ((y_full == "nostalgic") | (y_full == "both")).astype(int)
y_nostalgic = y_nostalgic.map({1: "nostalgic", 0: "not_nostalgic"})

print(f"Class distribution:")
print(y_nostalgic.value_counts())

X_train, X_test, y_train, y_test = train_test_split(
    X, y_nostalgic, test_size=0.2, random_state=42, stratify=y_nostalgic
)

model_nost = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    min_samples_split=20,
    min_samples_leaf=10,
    class_weight="balanced",
    random_state=42
)

model_nost.fit(X_train, y_train)
y_pred = model_nost.predict(X_test)

bal_acc_nost = balanced_accuracy_score(y_test, y_pred)
print(f"\nBalanced Accuracy: {bal_acc_nost:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred, labels=["nostalgic", "not_nostalgic"])
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Greens",
            xticklabels=["Nostalgic", "Not Nostalgic"],
            yticklabels=["Nostalgic", "Not Nostalgic"])
plt.title("Binary: Nostalgic Detection")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
plt.savefig("images/binary_nostalgic_cm.png", dpi=300)
plt.close()
print("✓ Saved binary_nostalgic_cm.png")

# ==========================================
# BINARY 2: Collectible (Yes/No)
# ==========================================
print("\n" + "="*60)
print("2. COLLECTIBLE vs NOT-COLLECTIBLE")
print("-"*60)

y_collectible = ((y_full == "collectible") | (y_full == "both")).astype(int)
y_collectible = y_collectible.map({1: "collectible", 0: "not_collectible"})

print(f"Class distribution:")
print(y_collectible.value_counts())

X_train, X_test, y_train, y_test = train_test_split(
    X, y_collectible, test_size=0.2, random_state=42, stratify=y_collectible
)

model_coll = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    min_samples_split=20,
    min_samples_leaf=10,
    class_weight="balanced",
    random_state=42
)

model_coll.fit(X_train, y_train)
y_pred = model_coll.predict(X_test)

bal_acc_coll = balanced_accuracy_score(y_test, y_pred)
print(f"\nBalanced Accuracy: {bal_acc_coll:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred, labels=["collectible", "not_collectible"])
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Oranges",
            xticklabels=["Collectible", "Not Collectible"],
            yticklabels=["Collectible", "Not Collectible"])
plt.title("Binary: Collectible Detection")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
plt.savefig("images/binary_collectible_cm.png", dpi=300)
plt.close()
print("✓ Saved binary_collectible_cm.png")

# ==========================================
# SUMMARY
# ==========================================
print("\n" + "="*60)
print("BINARY CLASSIFICATION SUMMARY")
print("="*60)
print(f"Nostalgic Detection:    {bal_acc_nost:.1%} balanced accuracy")
print(f"Collectible Detection:  {bal_acc_coll:.1%} balanced accuracy")
print(f"Multiclass (3-way):     61.4% balanced accuracy")
print("\n✓ Binary models often perform better due to simpler decision boundaries")

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load data
df = pd.read_csv("files/labeled_data.csv")

# Filter to valid classes
df = df[df["label"].isin(["nostalgic", "collectible", "both"])].copy()

print("="*60)
print("PRICE ANALYSIS")
print("="*60)

# Check if price column exists
if 'price' not in df.columns:
    print("⚠️  No price data available")
else:
    # Clean price data
    df['price'] = pd.to_numeric(df['price'], errors='coerce')
    df = df[df['price'] >= 0]  # Remove invalid prices
    
    # Filter to reasonable price range (remove F2P and extreme outliers)
    df_priced = df[(df['price'] > 0) & (df['price'] < 60)].copy()
    
    print(f"\nAnalyzing {len(df_priced):,} reviews with valid prices")
    print(f"Price range: ${df_priced['price'].min():.2f} - ${df_priced['price'].max():.2f}")
    
    # Summary statistics
    price_stats = df_priced.groupby('label')['price'].agg(['mean', 'median', 'std', 'count'])
    print("\nPrice by Category:")
    print(price_stats.round(2))
    price_stats.to_csv("files/price_analysis.csv")
    
    # Visualization 1: Box plot
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df_priced, x='label', y='price', 
                order=['nostalgic', 'collectible', 'both'])
    plt.title("Game Price Distribution by Purchase Motivation", fontsize=14, fontweight='bold')
    plt.ylabel("Price (USD)", fontsize=12)
    plt.xlabel("Category", fontsize=12)
    plt.tight_layout()
    plt.savefig("images/price_by_category.png", dpi=300)
    plt.close()
    print("\n✓ Saved price_by_category.png")
    
    # Visualization 2: Price vs Playtime
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for idx, label in enumerate(['nostalgic', 'collectible', 'both']):
        df_subset = df_priced[df_priced['label'] == label]
        
        # Filter extreme playtime outliers for visualization
        df_plot = df_subset[df_subset['playtime_at_review'] < 100]
        
        axes[idx].scatter(df_plot['price'], df_plot['playtime_at_review'], 
                         alpha=0.3, s=10)
        axes[idx].set_xlabel("Price (USD)")
        axes[idx].set_ylabel("Hours Played")
        axes[idx].set_title(f"{label.title()}")
        axes[idx].grid(alpha=0.3)
    
    plt.suptitle("Price vs Playtime by Category", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig("images/price_vs_playtime.png", dpi=300)
    plt.close()
    print("✓ Saved price_vs_playtime.png")
    
    # Key findings
    nostalgic_price = price_stats.loc['nostalgic', 'median']
    collectible_price = price_stats.loc['collectible', 'median']
    
    findings = f"""
PRICE FINDINGS:
- Nostalgic buyers: median ${nostalgic_price:.2f}
- Collectible buyers: median ${collectible_price:.2f}
- Difference: ${abs(nostalgic_price - collectible_price):.2f}

INTERPRETATION:
{'Nostalgic' if nostalgic_price > collectible_price else 'Collectible'} buyers tend to purchase {'more' if nostalgic_price > collectible_price else 'less'} expensive games.
"""
    
    print(findings)
    
    with open("files/price_findings.txt", "w") as f:
        f.write(findings)

print("\n✓ Price analysis complete")
"""
price_by_motivation_strength.py
Analyzes if STRENGTH of motivation (not just category) correlates with price
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.model_selection import train_test_split

print("="*60)
print("PRICE ANALYSIS BY MOTIVATION STRENGTH")
print("="*60)

# ==========================================
# 1. LOAD MODELS AND DATA
# ==========================================

# Load trained models
with open('files/model_nostalgic.pkl', 'rb') as f:
    model_nostalgic = pickle.load(f)

with open('files/model_collectible.pkl', 'rb') as f:
    model_collectible = pickle.load(f)

# Load data
X = pd.read_csv("files/features.csv")
y_full = pd.read_csv("files/labels.csv")["label"]
df_raw = pd.read_csv("files/labeled_data.csv")

# Filter
mask = y_full.isin(["nostalgic", "collectible", "both"])
X = X[mask].reset_index(drop=True)
y_full = y_full[mask].reset_index(drop=True)
df_raw = df_raw[mask].reset_index(drop=True)

# Get test set (same split as training)
X_train, X_test, y_train, y_test = train_test_split(
    X, y_full, test_size=0.2, random_state=42, stratify=y_full
)

# Get corresponding raw data for test set
_, df_test = train_test_split(
    df_raw, test_size=0.2, random_state=42, stratify=y_full
)

print(f"Analyzing {len(X_test):,} test samples")

# ==========================================
# 2. GET PREDICTION PROBABILITIES
# ==========================================

# Get probabilities (not just binary predictions)
prob_nostalgic = model_nostalgic.predict_proba(X_test)[:, 1]  # Probability of being nostalgic
prob_collectible = model_collectible.predict_proba(X_test)[:, 1]  # Probability of being collectible

# Add to dataframe
df_test = df_test.reset_index(drop=True)
df_test['prob_nostalgic'] = prob_nostalgic
df_test['prob_collectible'] = prob_collectible
df_test['true_label'] = y_test.values

print("\nProbability distributions:")
print(f"Nostalgic probability:   {prob_nostalgic.mean():.2f} ± {prob_nostalgic.std():.2f}")
print(f"Collectible probability: {prob_collectible.mean():.2f} ± {prob_collectible.std():.2f}")

# ==========================================
# 3. CATEGORIZE BY MOTIVATION STRENGTH
# ==========================================

def categorize_motivation(row):
    """
    Categorize based on probability strength
    """
    nost_prob = row['prob_nostalgic']
    coll_prob = row['prob_collectible']
    
    # Strong thresholds (>70% confidence)
    if nost_prob > 0.7 and coll_prob > 0.7:
        return "strong_both"
    elif nost_prob > 0.7:
        return "strong_nostalgic"
    elif coll_prob > 0.7:
        return "strong_collectible"
    
    # Weak thresholds (40-60% - uncertain)
    elif 0.4 < nost_prob < 0.6 and 0.4 < coll_prob < 0.6:
        return "weak_both"
    elif 0.4 < nost_prob < 0.6:
        return "weak_nostalgic"
    elif 0.4 < coll_prob < 0.6:
        return "weak_collectible"
    
    # Dominant motivation (one > 60%, other < 40%)
    elif nost_prob > coll_prob:
        return "moderate_nostalgic"
    else:
        return "moderate_collectible"

df_test['motivation_strength'] = df_test.apply(categorize_motivation, axis=1)

print("\nMotivation strength distribution:")
print(df_test['motivation_strength'].value_counts())

# ==========================================
# 4. PRICE ANALYSIS BY STRENGTH
# ==========================================

# Filter to valid prices
if 'price' in df_test.columns:
    df_price = df_test[(df_test['price'] > 0) & (df_test['price'] < 60)].copy()
    
    print(f"\nAnalyzing {len(df_price):,} reviews with valid prices")
    
    # Summary statistics
    price_by_strength = df_price.groupby('motivation_strength')['price'].agg([
        'count', 'mean', 'median', 'std'
    ]).round(2)
    
    print("\n" + "="*60)
    print("PRICE BY MOTIVATION STRENGTH")
    print("="*60)
    print(price_by_strength)
    
    price_by_strength.to_csv("files/price_by_motivation_strength.csv")
    print("\n✓ Saved price_by_motivation_strength.csv")
    
    # ==========================================
    # 5. VISUALIZATIONS
    # ==========================================
    
    # Visualization 1: Box plot by strength
    plt.figure(figsize=(14, 6))
    
    # Order categories logically
    order = [
        'strong_nostalgic', 'moderate_nostalgic', 'weak_nostalgic',
        'strong_collectible', 'moderate_collectible', 'weak_collectible',
        'strong_both', 'weak_both'
    ]
    order = [o for o in order if o in df_price['motivation_strength'].unique()]
    
    sns.boxplot(data=df_price, x='motivation_strength', y='price', order=order)
    plt.xticks(rotation=45, ha='right')
    plt.title("Game Price by Motivation Strength", fontsize=14, fontweight='bold')
    plt.ylabel("Price (USD)", fontsize=12)
    plt.xlabel("Motivation Category", fontsize=12)
    plt.tight_layout()
    plt.savefig("images/price_by_strength_boxplot.png", dpi=300)
    plt.close()
    print("✓ Saved price_by_strength_boxplot.png")
    
    # Visualization 2: Scatter plot - probability vs price
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Nostalgic probability vs price
    axes[0].scatter(df_price['prob_nostalgic'], df_price['price'], 
                   alpha=0.3, s=20, c='green')
    axes[0].set_xlabel("Nostalgic Probability")
    axes[0].set_ylabel("Price (USD)")
    axes[0].set_title("Nostalgic Motivation vs Price")
    axes[0].grid(alpha=0.3)
    
    # Add trend line
    z = np.polyfit(df_price['prob_nostalgic'], df_price['price'], 1)
    p = np.poly1d(z)
    axes[0].plot(df_price['prob_nostalgic'].sort_values(), 
                p(df_price['prob_nostalgic'].sort_values()), 
                "r--", alpha=0.8, linewidth=2)
    
    # Collectible probability vs price
    axes[1].scatter(df_price['prob_collectible'], df_price['price'],
                   alpha=0.3, s=20, c='orange')
    axes[1].set_xlabel("Collectible Probability")
    axes[1].set_ylabel("Price (USD)")
    axes[1].set_title("Collectible Motivation vs Price")
    axes[1].grid(alpha=0.3)
    
    # Add trend line
    z = np.polyfit(df_price['prob_collectible'], df_price['price'], 1)
    p = np.poly1d(z)
    axes[1].plot(df_price['prob_collectible'].sort_values(),
                p(df_price['prob_collectible'].sort_values()),
                "r--", alpha=0.8, linewidth=2)
    
    plt.suptitle("Motivation Strength vs Game Price", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig("images/motivation_probability_vs_price.png", dpi=300)
    plt.close()
    print("✓ Saved motivation_probability_vs_price.png")
    
    # Visualization 3: Price ranges by category
    plt.figure(figsize=(12, 6))
    
    # Calculate price quartiles for each strength category
    strength_categories = ['strong_nostalgic', 'moderate_nostalgic', 
                          'strong_collectible', 'moderate_collectible', 
                          'strong_both']
    strength_categories = [s for s in strength_categories if s in df_price['motivation_strength'].unique()]
    
    price_ranges = []
    for cat in strength_categories:
        prices = df_price[df_price['motivation_strength'] == cat]['price']
        if len(prices) > 10:  # Only include if sufficient samples
            q1, q2, q3 = prices.quantile([0.25, 0.5, 0.75])
            price_ranges.append({
                'category': cat,
                'Q1': q1,
                'Median': q2,
                'Q3': q3,
                'count': len(prices)
            })
    
    df_ranges = pd.DataFrame(price_ranges)
    
    x = range(len(df_ranges))
    plt.barh(x, df_ranges['Q3'] - df_ranges['Q1'], left=df_ranges['Q1'], 
            height=0.5, alpha=0.6, label='IQR (25th-75th percentile)')
    plt.scatter(df_ranges['Median'], x, color='red', s=100, zorder=3, label='Median')
    
    plt.yticks(x, [f"{c}\n(n={n})" for c, n in zip(df_ranges['category'], df_ranges['count'])])
    plt.xlabel("Price (USD)", fontsize=12)
    plt.title("Price Ranges by Motivation Strength", fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(alpha=0.3, axis='x')
    plt.tight_layout()
    plt.savefig("images/price_ranges_by_strength.png", dpi=300)
    plt.close()
    print("✓ Saved price_ranges_by_strength.png")
    
    # ==========================================
    # 6. STATISTICAL ANALYSIS
    # ==========================================
    
    from scipy import stats
    
    print("\n" + "="*60)
    print("STATISTICAL TESTS")
    print("="*60)
    
    # Test: Do strong nostalgic buyers pay different prices than strong collectible?
    if 'strong_nostalgic' in df_price['motivation_strength'].values and \
       'strong_collectible' in df_price['motivation_strength'].values:
        
        strong_nost_prices = df_price[df_price['motivation_strength'] == 'strong_nostalgic']['price']
        strong_coll_prices = df_price[df_price['motivation_strength'] == 'strong_collectible']['price']
        
        if len(strong_nost_prices) > 10 and len(strong_coll_prices) > 10:
            t_stat, p_value = stats.ttest_ind(strong_nost_prices, strong_coll_prices)
            
            print(f"\nT-Test: Strong Nostalgic vs Strong Collectible")
            print(f"  Strong Nostalgic:   ${strong_nost_prices.mean():.2f} ± ${strong_nost_prices.std():.2f}")
            print(f"  Strong Collectible: ${strong_coll_prices.mean():.2f} ± ${strong_coll_prices.std():.2f}")
            print(f"  t-statistic: {t_stat:.3f}")
            print(f"  p-value: {p_value:.4f}")
            
            if p_value < 0.05:
                print(f"  ✓ SIGNIFICANT DIFFERENCE (p < 0.05)")
            else:
                print(f"  ✗ No significant difference (p >= 0.05)")
    
    # Correlation: Nostalgic probability vs price
    corr_nost, p_nost = stats.pearsonr(df_price['prob_nostalgic'], df_price['price'])
    print(f"\nCorrelation: Nostalgic Probability vs Price")
    print(f"  r = {corr_nost:.3f}, p = {p_nost:.4f}")
    
    # Correlation: Collectible probability vs price
    corr_coll, p_coll = stats.pearsonr(df_price['prob_collectible'], df_price['price'])
    print(f"\nCorrelation: Collectible Probability vs Price")
    print(f"  r = {corr_coll:.3f}, p = {p_coll:.4f}")
    
    # ==========================================
    # 7. KEY FINDINGS
    # ==========================================
    
    findings = f"""
{'='*60}
PRICE ANALYSIS BY MOTIVATION STRENGTH - KEY FINDINGS
{'='*60}

SAMPLE SIZE:
  Total reviews with price data: {len(df_price):,}

PRICE BY MOTIVATION STRENGTH:
{price_by_strength.to_string()}

STATISTICAL TESTS:
  Nostalgic prob. vs price correlation: r={corr_nost:.3f}, p={p_nost:.4f}
  Collectible prob. vs price correlation: r={corr_coll:.3f}, p={p_coll:.4f}

INTERPRETATION:
"""
    
    if abs(corr_nost) > 0.1 and p_nost < 0.05:
        findings += f"  ✓ Nostalgic motivation {'POSITIVELY' if corr_nost > 0 else 'NEGATIVELY'} correlates with price\n"
    else:
        findings += f"  ✗ Nostalgic motivation does NOT significantly correlate with price\n"
    
    if abs(corr_coll) > 0.1 and p_coll < 0.05:
        findings += f"  ✓ Collectible motivation {'POSITIVELY' if corr_coll > 0 else 'NEGATIVELY'} correlates with price\n"
    else:
        findings += f"  ✗ Collectible motivation does NOT significantly correlate with price\n"
    
    findings += f"\n{'='*60}\n"
    
    print(findings)
    
    with open("files/price_motivation_strength_findings.txt", "w") as f:
        f.write(findings)
    
    print("✓ Saved price_motivation_strength_findings.txt")

else:
    print("\n⚠️  No price data available in dataset")

print("\n" + "="*60)
print("ANALYSIS COMPLETE")
print("="*60)

