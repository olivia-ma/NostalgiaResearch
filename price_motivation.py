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
with open('model_nostalgic.pkl', 'rb') as f:
    model_nostalgic = pickle.load(f)

with open('model_collectible.pkl', 'rb') as f:
    model_collectible = pickle.load(f)

# Load data
X = pd.read_csv("features.csv")
y_full = pd.read_csv("labels.csv")["label"]
df_raw = pd.read_csv("labeled_data.csv")

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
    
    price_by_strength.to_csv("price_by_motivation_strength.csv")
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
    plt.savefig("price_by_strength_boxplot.png", dpi=300)
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
    plt.savefig("motivation_probability_vs_price.png", dpi=300)
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
    plt.savefig("price_ranges_by_strength.png", dpi=300)
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
    
    with open("price_motivation_strength_findings.txt", "w") as f:
        f.write(findings)
    
    print("✓ Saved price_motivation_strength_findings.txt")

else:
    print("\n⚠️  No price data available in dataset")

print("\n" + "="*60)
print("ANALYSIS COMPLETE")
print("="*60)
