import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    balanced_accuracy_score,
    f1_score
)
import matplotlib.pyplot as plt
import seaborn as sns

# ------------------------------
# 1. LOAD CLEAN FEATURES + LABELS
# ------------------------------

X = pd.read_csv("files/features.csv")
y = pd.read_csv("files/labels.csv")["label"]

# Ensure classes match your pipeline
VALID_CLASSES = ["nostalgic", "collectible", "both"]

mask = y.isin(VALID_CLASSES)
X = X[mask].reset_index(drop=True)
y = y[mask].reset_index(drop=True)

print("="*60)
print("CLASS DISTRIBUTION")
print("="*60)
print(y.value_counts())
print(f"\nTotal samples: {len(y):,}")
print()

# Check for data leakage signals
if 'has_nostalgia_keywords' in X.columns or 'has_collectible_keywords' in X.columns:
    print("⚠️  WARNING: Keyword flags detected in features - potential data leakage!")
else:
    print("✓ No keyword flags in features - good!")

# ------------------------------
# 2. TRAIN/TEST SPLIT
# ------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print(f"\nTrain size: {len(X_train):,}")
print(f"Test size:  {len(X_test):,}")

# ------------------------------
# 3. MODEL TRAINING + EVALUATION
# ------------------------------

# Try multiple models
models = {
    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        max_depth=10,  # Limit depth to prevent overfitting
        min_samples_split=20,
        min_samples_leaf=10,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    ),
    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=42
    )
}

results = {}

for name, model in models.items():
    print("\n" + "="*60)
    print(f"TRAINING: {name}")
    print("="*60)
    
    # Train
    model.fit(X_train, y_train)
    
    # Predict
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    # Metrics
    train_acc = accuracy_score(y_train, y_pred_train)
    test_acc = accuracy_score(y_test, y_pred_test)
    test_balanced_acc = balanced_accuracy_score(y_test, y_pred_test)
    test_f1_macro = f1_score(y_test, y_pred_test, average='macro')
    
    results[name] = {
        "train_accuracy": train_acc,
        "test_accuracy": test_acc,
        "balanced_accuracy": test_balanced_acc,
        "f1_macro": test_f1_macro
    }
    
    print(f"\nTrain Accuracy:     {train_acc:.4f}")
    print(f"Test Accuracy:      {test_acc:.4f}")
    print(f"Balanced Accuracy:  {test_balanced_acc:.4f}")
    print(f"F1 Score (Macro):   {test_f1_macro:.4f}")
    
    # Check for overfitting
    if train_acc > 0.95 and test_acc > 0.95:
        print("\n⚠️  POSSIBLE DATA LEAKAGE - Both train and test accuracy > 95%")
    elif train_acc - test_acc > 0.15:
        print("\n⚠️  OVERFITTING DETECTED - Train accuracy much higher than test")
    else:
        print("\n✓ Model performance looks reasonable")
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred_test))
    
    # Cross-validation to verify
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='balanced_accuracy')
    print(f"\n5-Fold CV Balanced Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")

# ------------------------------
# 4. SELECT BEST MODEL & SAVE ARTIFACTS
# ------------------------------

best_model_name = max(results, key=lambda k: results[k]['f1_macro'])
best_model = models[best_model_name]

print("\n" + "="*60)
print(f"BEST MODEL: {best_model_name}")
print("="*60)

# Retrain on full training set
best_model.fit(X_train, y_train)
y_pred = best_model.predict(X_test)

# Confusion Matrix
labels = VALID_CLASSES
cm = confusion_matrix(y_test, y_pred, labels=labels)

plt.figure(figsize=(8, 6))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=labels,
    yticklabels=labels,
    cbar_kws={'label': 'Count'}
)
plt.title(f"Confusion Matrix - {best_model_name}")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
plt.savefig("images/confusion_matrix.png", dpi=300)
plt.close()
print("\n✓ Saved confusion_matrix.png")

# Feature Importance (if available)
if hasattr(best_model, 'feature_importances_'):
    feat_imp = pd.Series(
        best_model.feature_importances_,
        index=X.columns
    ).sort_values(ascending=False).head(25)
    
    print("\nTop 25 Features:")
    print(feat_imp)
    
    feat_imp.to_csv("files/feature_importance.csv")
    
    plt.figure(figsize=(10, 8))
    feat_imp[::-1].plot(kind="barh")
    plt.title(f"Top 25 Most Important Features - {best_model_name}")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig("images/feature_importance.png", dpi=300)
    plt.close()
    print("✓ Saved feature_importance.csv and feature_importance.png")

elif hasattr(best_model, 'coef_'):
    # For linear models like LogisticRegression
    coef_df = pd.DataFrame(
        best_model.coef_,
        columns=X.columns,
        index=best_model.classes_
    ).T
    
    # Get absolute max coefficient for each feature
    coef_df['max_abs'] = coef_df.abs().max(axis=1)
    top_features = coef_df.nlargest(25, 'max_abs')
    
    print("\nTop 25 Features by Coefficient Magnitude:")
    print(top_features)
    
    top_features.to_csv("files/feature_importance.csv")
    print("✓ Saved feature_importance.csv")

# Save results summary
results_df = pd.DataFrame(results).T
results_df.to_csv("files/model_comparison.csv")
print("\n✓ Saved model_comparison.csv")

print("\n" + "="*60)
print("TRAINING COMPLETE")
print("="*60)