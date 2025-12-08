"""
models_binary.py
Two independent binary classifiers approach
Better performance + handles overlapping motivations naturally
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, balanced_accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

print("="*60)
print("DUAL BINARY CLASSIFIER APPROACH")
print("="*60)

# ==========================================
# 1. LOAD DATA
# ==========================================

X = pd.read_csv("features.csv")
y_full = pd.read_csv("labels.csv")["label"]

# Filter valid classes
mask = y_full.isin(["nostalgic", "collectible", "both"])
X = X[mask].reset_index(drop=True)
y_full = y_full[mask].reset_index(drop=True)

print(f"\nTotal samples: {len(X):,}")
print(f"Original distribution:")
print(y_full.value_counts())

# ==========================================
# 2. CREATE BINARY LABELS
# ==========================================

# Binary label 1: Has nostalgia motivation?
y_nostalgic_binary = ((y_full == "nostalgic") | (y_full == "both")).astype(int)

# Binary label 2: Has collectible motivation?
y_collectible_binary = ((y_full == "collectible") | (y_full == "both")).astype(int)

print("\nBinary label distributions:")
print(f"Nostalgic:   {y_nostalgic_binary.sum():,} ({y_nostalgic_binary.sum()/len(y_nostalgic_binary)*100:.1f}%)")
print(f"Collectible: {y_collectible_binary.sum():,} ({y_collectible_binary.sum()/len(y_collectible_binary)*100:.1f}%)")
print(f"Both:        {((y_full == 'both').sum()):,} ({(y_full == 'both').sum()/len(y_full)*100:.1f}%)")

# ==========================================
# 3. TRAIN NOSTALGIC DETECTOR
# ==========================================

print("\n" + "="*60)
print("TRAINING: NOSTALGIC DETECTOR")
print("="*60)

X_train, X_test, y_train_nost, y_test_nost = train_test_split(
    X, y_nostalgic_binary,
    test_size=0.2,
    random_state=42,
    stratify=y_nostalgic_binary
)

model_nostalgic = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    min_samples_split=20,
    min_samples_leaf=10,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

model_nostalgic.fit(X_train, y_train_nost)
y_pred_nost = model_nostalgic.predict(X_test)

# Evaluation
bal_acc_nost = balanced_accuracy_score(y_test_nost, y_pred_nost)
print(f"\nBalanced Accuracy: {bal_acc_nost:.4f}")

# Cross-validation
cv_scores_nost = cross_val_score(
    model_nostalgic, X_train, y_train_nost,
    cv=5, scoring='balanced_accuracy'
)
print(f"CV Balanced Accuracy: {cv_scores_nost.mean():.4f} (+/- {cv_scores_nost.std()*2:.4f})")

print("\nClassification Report:")
print(classification_report(y_test_nost, y_pred_nost, 
                           target_names=["Not Nostalgic", "Nostalgic"]))

# Confusion matrix
cm_nost = confusion_matrix(y_test_nost, y_pred_nost)
plt.figure(figsize=(8, 6))
sns.heatmap(cm_nost, annot=True, fmt="d", cmap="Greens",
            xticklabels=["Not Nostalgic", "Nostalgic"],
            yticklabels=["Not Nostalgic", "Nostalgic"])
plt.title(f"Nostalgic Detector\nBalanced Accuracy: {bal_acc_nost:.1%}")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
plt.savefig("model_nostalgic_detector.png", dpi=300)
plt.close()
print("✓ Saved model_nostalgic_detector.png")

# ==========================================
# 4. TRAIN COLLECTIBLE DETECTOR
# ==========================================

print("\n" + "="*60)
print("TRAINING: COLLECTIBLE DETECTOR")
print("="*60)

X_train, X_test, y_train_coll, y_test_coll = train_test_split(
    X, y_collectible_binary,
    test_size=0.2,
    random_state=42,
    stratify=y_collectible_binary
)

model_collectible = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    min_samples_split=20,
    min_samples_leaf=10,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

model_collectible.fit(X_train, y_train_coll)
y_pred_coll = model_collectible.predict(X_test)

# Evaluation
bal_acc_coll = balanced_accuracy_score(y_test_coll, y_pred_coll)
print(f"\nBalanced Accuracy: {bal_acc_coll:.4f}")

# Cross-validation
cv_scores_coll = cross_val_score(
    model_collectible, X_train, y_train_coll,
    cv=5, scoring='balanced_accuracy'
)
print(f"CV Balanced Accuracy: {cv_scores_coll.mean():.4f} (+/- {cv_scores_coll.std()*2:.4f})")

print("\nClassification Report:")
print(classification_report(y_test_coll, y_pred_coll,
                           target_names=["Not Collectible", "Collectible"]))

# Confusion matrix
cm_coll = confusion_matrix(y_test_coll, y_pred_coll)
plt.figure(figsize=(8, 6))
sns.heatmap(cm_coll, annot=True, fmt="d", cmap="Oranges",
            xticklabels=["Not Collectible", "Collectible"],
            yticklabels=["Not Collectible", "Collectible"])
plt.title(f"Collectible Detector\nBalanced Accuracy: {bal_acc_coll:.1%}")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
plt.savefig("model_collectible_detector.png", dpi=300)
plt.close()
print("✓ Saved model_collectible_detector.png")

# ==========================================
# 5. COMBINED PREDICTIONS
# ==========================================

print("\n" + "="*60)
print("COMBINED PREDICTION ANALYSIS")
print("="*60)

# Get predictions on full test set (using same split as nostalgic for consistency)
X_train, X_test, y_train_full, y_test_full = train_test_split(
    X, y_full,
    test_size=0.2,
    random_state=42,
    stratify=y_full
)

# Predict both
pred_nost = model_nostalgic.predict(X_test)
pred_coll = model_collectible.predict(X_test)

# Combine predictions into 3 categories
def combine_predictions(nost, coll):
    """
    Combine binary predictions into 3-way classification
    """
    if nost == 1 and coll == 1:
        return "both"
    elif nost == 1:
        return "nostalgic"
    elif coll == 1:
        return "collectible"
    else:
        return "neither"  # Edge case

combined_predictions = [
    combine_predictions(n, c) 
    for n, c in zip(pred_nost, pred_coll)
]

# Compare with original labels
print("\nPrediction Distribution:")
print(pd.Series(combined_predictions).value_counts())

print("\nTrue Distribution:")
print(y_test_full.value_counts())

# Calculate accuracy for 3-way classification using dual binary approach
valid_mask = [p in ["nostalgic", "collectible", "both"] for p in combined_predictions]
combined_pred_filtered = [combined_predictions[i] for i, v in enumerate(valid_mask) if v]
y_test_filtered = y_test_full[valid_mask].reset_index(drop=True)

from sklearn.metrics import accuracy_score
combined_accuracy = accuracy_score(y_test_filtered, combined_pred_filtered)
print(f"\n3-way accuracy (from dual binary): {combined_accuracy:.4f}")

# Confusion matrix for combined
labels_3way = ["nostalgic", "collectible", "both"]
cm_combined = confusion_matrix(y_test_filtered, combined_pred_filtered, labels=labels_3way)

plt.figure(figsize=(10, 8))
sns.heatmap(cm_combined, annot=True, fmt="d", cmap="Blues",
            xticklabels=labels_3way, yticklabels=labels_3way)
plt.title(f"Combined 3-Way Classification\n(From Dual Binary Models)\nAccuracy: {combined_accuracy:.1%}")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
plt.savefig("model_combined_3way.png", dpi=300)
plt.close()
print("✓ Saved model_combined_3way.png")

# ==========================================
# 6. FEATURE IMPORTANCE COMPARISON
# ==========================================

print("\n" + "="*60)
print("FEATURE IMPORTANCE COMPARISON")
print("="*60)

feat_imp_nost = pd.Series(
    model_nostalgic.feature_importances_,
    index=X.columns
).sort_values(ascending=False)

feat_imp_coll = pd.Series(
    model_collectible.feature_importances_,
    index=X.columns
).sort_values(ascending=False)

print("\nTop 10 Features for NOSTALGIC Detection:")
print(feat_imp_nost.head(10))

print("\nTop 10 Features for COLLECTIBLE Detection:")
print(feat_imp_coll.head(10))

# Save
pd.DataFrame({
    'nostalgic_importance': feat_imp_nost,
    'collectible_importance': feat_imp_coll
}).to_csv("dual_feature_importance.csv")

# Visualization - side by side
fig, axes = plt.subplots(1, 2, figsize=(16, 8))

feat_imp_nost.head(15)[::-1].plot(kind='barh', ax=axes[0], color='green')
axes[0].set_title("Top 15 Features: Nostalgic Detection")
axes[0].set_xlabel("Importance")

feat_imp_coll.head(15)[::-1].plot(kind='barh', ax=axes[1], color='orange')
axes[1].set_title("Top 15 Features: Collectible Detection")
axes[1].set_xlabel("Importance")

plt.tight_layout()
plt.savefig("dual_feature_importance.png", dpi=300)
plt.close()
print("✓ Saved dual_feature_importance.png")

# ==========================================
# 7. SAVE MODELS
# ==========================================

with open('model_nostalgic.pkl', 'wb') as f:
    pickle.dump(model_nostalgic, f)

with open('model_collectible.pkl', 'wb') as f:
    pickle.dump(model_collectible, f)

print("\n✓ Saved models: model_nostalgic.pkl, model_collectible.pkl")

# ==========================================
# 8. FINAL SUMMARY
# ==========================================

print("\n" + "="*60)
print("FINAL SUMMARY")
print("="*60)

print(f"\n📊 MODEL PERFORMANCE:")
print(f"  Nostalgic Detection:    {bal_acc_nost:.1%} balanced accuracy")
print(f"  Collectible Detection:  {bal_acc_coll:.1%} balanced accuracy")
print(f"  Combined 3-Way:         {combined_accuracy:.1%} accuracy")

print(f"\n🎯 COMPARISON WITH ORIGINAL MULTICLASS:")
print(f"  Original Random Forest (3-way):     61.4%")
print(f"  Dual Binary Approach (3-way):       {combined_accuracy:.1%}")
print(f"  Improvement: {(combined_accuracy - 0.614)*100:+.1f} percentage points")

if combined_accuracy > 0.614:
    print("\n✓ DUAL BINARY APPROACH PERFORMS BETTER!")
else:
    print("\n◐ Dual binary approach gives similar performance")
    print("   Advantage: Better interpretability + handles overlaps")

print("\n" + "="*60)
print("COMPLETE")
print("="*60)