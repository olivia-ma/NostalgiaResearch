import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, balanced_accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
X = pd.read_csv("features.csv")
y_full = pd.read_csv("labels.csv")["label"]
df_raw = pd.read_csv("labeled_data.csv")

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
plt.savefig("binary_nostalgic_cm.png", dpi=300)
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
plt.savefig("binary_collectible_cm.png", dpi=300)
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
df = pd.read_csv("labeled_data.csv")

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
    price_stats.to_csv("price_analysis.csv")
    
    # Visualization 1: Box plot
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df_priced, x='label', y='price', 
                order=['nostalgic', 'collectible', 'both'])
    plt.title("Game Price Distribution by Purchase Motivation", fontsize=14, fontweight='bold')
    plt.ylabel("Price (USD)", fontsize=12)
    plt.xlabel("Category", fontsize=12)
    plt.tight_layout()
    plt.savefig("price_by_category.png", dpi=300)
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
    plt.savefig("price_vs_playtime.png", dpi=300)
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
    
    with open("price_findings.txt", "w") as f:
        f.write(findings)

print("\n✓ Price analysis complete")
