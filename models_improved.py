"""
models_best.py
Conservative approach that prevents overfitting to synthetic data
Strategy: Mild undersampling + careful ensemble
Expected: 65-70% balanced accuracy (sustainable)
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, balanced_accuracy_score
from sklearn.utils import resample
import matplotlib.pyplot as plt
import seaborn as sns

# ------------------------------
# 1. LOAD DATA
# ------------------------------

X = pd.read_csv("features.csv")
y = pd.read_csv("labels.csv")["label"]
df_raw = pd.read_csv("labeled_data.csv")

VALID_CLASSES = ["nostalgic", "collectible", "both"]
mask = y.isin(VALID_CLASSES)

X = X[mask].reset_index(drop=True)
y = y[mask].reset_index(drop=True)
df_raw = df_raw[mask].reset_index(drop=True)

print("="*60)
print("CONSERVATIVE IMPROVED MODEL")
print("="*60)
print("\nOriginal class distribution:")
print(y.value_counts())
print()

# ------------------------------
# 2. ADD DOMAIN FEATURES
# ------------------------------

print("Adding domain-specific features...")

# Calculate if not present
if 'review_length' not in df_raw.columns:
    df_raw['review_length'] = df_raw['review_text'].fillna("").str.split().str.len()

# Engagement tiers (using raw values from df_raw)
X['low_playtime_flag'] = (df_raw['playtime_at_review'] < 2).astype(int)
X['high_playtime_flag'] = (df_raw['playtime_at_review'] > 20).astype(int)

# Game age brackets (using raw values)
X['very_old_game_flag'] = (df_raw['game_age_at_review'] > 8).astype(int)
X['recent_game_flag'] = (df_raw['game_age_at_review'] < 3).astype(int)

# Review sophistication
X['very_sophisticated'] = (X['very_long_review'] == 1).astype(int)

# Interaction features
X['positive_high_playtime'] = X['sentiment'] * X['high_playtime_flag']
X['positive_old_game'] = X['sentiment'] * X['very_old_game_flag']

print(f"Enhanced feature count: {X.shape[1]}")

# ------------------------------
# 3. CONSERVATIVE BALANCING
# ------------------------------

print("\nApplying conservative undersampling...")

# Combine X and y for sampling
df_combined = pd.concat([X, y.rename('label')], axis=1)

# Separate by class
nostalgic = df_combined[df_combined['label'] == 'nostalgic']
both = df_combined[df_combined['label'] == 'both']
collectible = df_combined[df_combined['label'] == 'collectible']

print(f"\nOriginal counts:")
print(f"  Nostalgic:   {len(nostalgic):,}")
print(f"  Both:        {len(both):,}")
print(f"  Collectible: {len(collectible):,}")

# Conservative strategy: downsample collectible to 3x minority
target_collectible = len(nostalgic) * 3

collectible_downsampled = resample(
    collectible,
    n_samples=target_collectible,
    random_state=42,
    replace=False
)

# Combine
df_balanced = pd.concat([nostalgic, both, collectible_downsampled])
df_balanced = df_balanced.sample(frac=1, random_state=42)

print(f"\nBalanced counts:")
print(df_balanced['label'].value_counts())
print(f"Total: {len(df_balanced):,}")

# Split back into X and y
y_balanced = df_balanced['label']
X_balanced = df_balanced.drop('label', axis=1)

# ------------------------------
# 4. TRAIN/TEST SPLIT
# ------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X_balanced, y_balanced,
    test_size=0.2,
    random_state=42,
    stratify=y_balanced
)

print(f"\nTrain size: {len(X_train):,}")
print(f"Test size:  {len(X_test):,}")

# ------------------------------
# 5. TRAIN MULTIPLE MODELS
# ------------------------------

print("\n" + "="*60)
print("TRAINING MODELS")
print("="*60)

models = {
    "Random Forest": RandomForestClassifier(
        n_estimators=300,
        max_depth=15,
        min_samples_split=15,
        min_samples_leaf=8,
        max_features='sqrt',
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    ),
    
    "Gradient Boosting": GradientBoostingClassifier(
        n_estimators=200,
        max_depth=8,
        learning_rate=0.1,
        min_samples_split=20,
        min_samples_leaf=10,
        random_state=42
    ),
    
    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        C=1.0,
        class_weight='balanced',
        random_state=42
    )
}

results = {}

for name, model in models.items():
    print(f"\n{name}:")
    
    # Train
    model.fit(X_train, y_train)
    
    # Predict
    y_pred = model.predict(X_test)
    
    # Metrics
    bal_acc = balanced_accuracy_score(y_test, y_pred)
    
    # Cross-validation
    cv_scores = cross_val_score(
        model, X_train, y_train,
        cv=5,
        scoring='balanced_accuracy'
    )
    
    results[name] = {
        'test_balanced_acc': bal_acc,
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std()
    }
    
    print(f"  Test Balanced Accuracy: {bal_acc:.4f}")
    print(f"  CV Balanced Accuracy:   {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")
    
    # Check if CV and test are similar (good sign)
    diff = abs(cv_scores.mean() - bal_acc)
    if diff < 0.03:
        print(f"  ✓ Good generalization (CV-test diff: {diff:.4f})")
    else:
        print(f"  ⚠ Possible overfitting (CV-test diff: {diff:.4f})")

# ------------------------------
# 6. SELECT BEST MODEL
# ------------------------------

best_model_name = max(results, key=lambda k: results[k]['test_balanced_acc'])
best_model = models[best_model_name]

print("\n" + "="*60)
print(f"BEST MODEL: {best_model_name}")
print("="*60)

# Retrain and evaluate
best_model.fit(X_train, y_train)
y_pred = best_model.predict(X_test)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

bal_acc = balanced_accuracy_score(y_test, y_pred)
print(f"\nBalanced Accuracy: {bal_acc:.4f}")

# ------------------------------
# 7. TEST ON ORIGINAL IMBALANCED TEST SET
# ------------------------------

print("\n" + "="*60)
print("VALIDATION ON ORIGINAL IMBALANCED DATA")
print("="*60)

# Get original unbalanced test set
X_orig_train, X_orig_test, y_orig_train, y_orig_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nOriginal test set distribution:")
print(y_orig_test.value_counts())

# Predict on original test set
y_orig_pred = best_model.predict(X_orig_test)

print("\nPerformance on original imbalanced test set:")
print(classification_report(y_orig_test, y_orig_pred))

orig_bal_acc = balanced_accuracy_score(y_orig_test, y_orig_pred)
print(f"\nBalanced Accuracy (original test): {orig_bal_acc:.4f}")

# ------------------------------
# 8. SAVE ARTIFACTS
# ------------------------------

# Confusion matrix on original test set
labels = VALID_CLASSES
cm = confusion_matrix(y_orig_test, y_orig_pred, labels=labels)

plt.figure(figsize=(10, 8))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=labels,
    yticklabels=labels
)
plt.title(f"Best Model: {best_model_name}\n(Tested on Original Imbalanced Data)")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
plt.savefig("confusion_matrix_best.png", dpi=300)
plt.close()
print("\n✓ Saved confusion_matrix_best.png")

# Feature importance
if hasattr(best_model, 'feature_importances_'):
    feat_imp = pd.Series(
        best_model.feature_importances_,
        index=X.columns
    ).sort_values(ascending=False).head(25)
    
    print("\nTop 25 Features:")
    print(feat_imp)
    
    feat_imp.to_csv("feature_importance_best.csv")
    
    plt.figure(figsize=(10, 8))
    feat_imp[::-1].plot(kind="barh", color='darkgreen')
    plt.title(f"Top 25 Features - {best_model_name}")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig("feature_importance_best.png", dpi=300)
    plt.close()
    print("✓ Saved feature_importance_best.csv and .png")

# ------------------------------
# 9. FINAL SUMMARY
# ------------------------------

print("\n" + "="*60)
print("FINAL RESULTS SUMMARY")
print("="*60)

print(f"\nBest Model: {best_model_name}")
print(f"Balanced test set accuracy:  {bal_acc:.4f}")
print(f"Original test set accuracy:  {orig_bal_acc:.4f}")
print(f"CV mean:                     {results[best_model_name]['cv_mean']:.4f}")

# Compare with baseline
baseline_rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    min_samples_split=20,
    min_samples_leaf=10,
    class_weight="balanced",
    random_state=42
)

baseline_rf.fit(X_orig_train, y_orig_train)
baseline_pred = baseline_rf.predict(X_orig_test)
baseline_acc = balanced_accuracy_score(y_orig_test, baseline_pred)

print(f"\nBaseline model:              {baseline_acc:.4f}")
print(f"Improvement:                 +{(orig_bal_acc - baseline_acc)*100:.1f} percentage points")

if orig_bal_acc >= 0.65:
    print("\n✓ GOOD RESULT: 65%+ balanced accuracy")
    if orig_bal_acc >= 0.70:
        print("✓✓ EXCELLENT: 70%+ achieved!")
else:
    print("\n◐ Modest improvement - consider:")
    print("  - Collecting more minority class samples")
    print("  - Manual label validation")
    print("  - More sophisticated text features")

print("\n" + "="*60)
print("COMPLETE")
print("="*60)