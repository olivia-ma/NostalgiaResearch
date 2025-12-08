import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load the labeled data (this has all original columns)
df = pd.read_csv("labeled_data.csv")

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
plt.savefig("insight_playtime.png", dpi=300, bbox_inches="tight")
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
plt.savefig("insight_game_age.png", dpi=300, bbox_inches="tight")
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

summary.to_csv("insights_summary.csv")
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
    plt.savefig("insight_genres.png", dpi=300)
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
plt.savefig("insight_review_length.png", dpi=300, bbox_inches="tight")
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
plt.savefig("insight_sentiment.png", dpi=300, bbox_inches="tight")
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

with open("key_findings.txt", "w") as f:
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