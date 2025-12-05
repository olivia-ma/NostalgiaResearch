import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df_raw = pd.read_csv("labeled_data.csv")
df_features = pd.read_csv("features.csv")
df = pd.concat([df_raw.reset_index(drop=True), df_features], axis=1)


VALID_CLASSES = ["nostalgic", "collectible", "both"]
df = df[df["label"].isin(VALID_CLASSES)]

# =========================
# INSIGHT 1 — Playtime
# =========================

plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x="label", y="playtime_at_review")
plt.yscale("log")
plt.title("Playtime Distribution by Category")
plt.ylabel("Hours Played (log scale)")
plt.savefig("insight_playtime.png", dpi=300, bbox_inches="tight")
plt.close()

# =========================
# INSIGHT 2 — Game Age
# =========================

plt.figure(figsize=(10, 6))
sns.violinplot(data=df, x="label", y="game_age_at_review")
plt.title("Game Age at Review")
plt.ylabel("Years Since Release")
plt.savefig("insight_game_age.png", dpi=300, bbox_inches="tight")
plt.close()

# =========================
# INSIGHT 3 — Summary Stats
# =========================

summary = df.groupby("label").agg({
    "playtime_at_review": ["mean", "median"],
    "game_age_at_review": ["mean", "median"],
    "review_length": "mean",
    "recommended": "mean"
}).round(2)

print("\n=== SUMMARY STATISTICS ===\n")
print(summary)
summary.to_csv("insights_summary.csv")

# =========================
# INSIGHT 4 — Keyword Flags
# =========================

keyword_stats = df.groupby("label").agg({
    "has_nostalgia_keywords": "mean",
    "has_collectible_keywords": "mean"
}).round(3)

print("\n=== KEYWORD PRESENCE ===\n")
print(keyword_stats)

# =========================
# INSIGHT 5 — Genre Breakdown
# =========================

if "genre" in df.columns:
    genre_counts = df.groupby(["label", "genre"]).size().unstack(fill_value=0)
    top_genres = genre_counts.sum().nlargest(10).index

    plt.figure(figsize=(12, 6))
    genre_counts[top_genres].T.plot(kind="bar")
    plt.title("Top Genres by Category")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig("insight_genres.png", dpi=300)
    plt.close()

# =========================
# KEY FINDINGS TEXT FILE
# =========================

findings = f"""
KEY FINDINGS:

1. Playtime:
   - Nostalgic median hours:     {summary.loc["nostalgic", ("playtime_at_review", "median")]:.2f}
   - Collectible median hours:   {summary.loc["collectible", ("playtime_at_review", "median")]:.2f}

2. Game Age:
   - Nostalgic avg age:          {summary.loc["nostalgic", ("game_age_at_review", "mean")]:.2f}
   - Collectible avg age:        {summary.loc["collectible", ("game_age_at_review", "mean")]:.2f}

3. Sentiment (Recommended %):
   - Nostalgic:                  {summary.loc["nostalgic", ("recommended", "mean")] * 100:.1f}%
   - Collectible:                {summary.loc["collectible", ("recommended", "mean")] * 100:.1f}%

4. Behavioral Signals:
   - Nostalgia buyers: long playtime, older games, positive sentiment.
   - Collectible buyers: low playtime, investment language, sealed/unplayed cues.
"""

with open("key_findings.txt", "w") as f:
    f.write(findings)

print(findings)
