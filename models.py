import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    balanced_accuracy_score
)
import matplotlib.pyplot as plt
import seaborn as sns

# ------------------------------
# 1. LOAD CLEAN FEATURES + LABELS
# ------------------------------

X = pd.read_csv("features.csv")
y = pd.read_csv("labels.csv")["label"]

# Ensure classes match your pipeline
VALID_CLASSES = ["nostalgic", "collectible", "both"]

mask = y.isin(VALID_CLASSES)
X = X[mask].reset_index(drop=True)
y = y[mask].reset_index(drop=True)

print("Class counts:")
print(y.value_counts(), "\n")

# ------------------------------
# 2. TRAIN/TEST SPLIT
# ------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ------------------------------
# 3. RANDOM FOREST MODEL
# ------------------------------

# model = RandomForestClassifier(
#     n_estimators=300,
#     max_depth=None,
#     class_weight="balanced",
#     random_state=42,
#     n_jobs=-1
# )

# model.fit(X_train, y_train)

from sklearn.svm import LinearSVC

model = LinearSVC()
model.fit(X_train, y_train)

preds = model.predict(X_test)
print(classification_report(y_test, preds))


# ------------------------------
# 4. PREDICTIONS + METRICS
# ------------------------------

y_pred = model.predict(X_test)

print("\n=== Classification Report ===")
print(classification_report(y_test, y_pred))

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Balanced Accuracy:", balanced_accuracy_score(y_test, y_pred))

# ------------------------------
# 5. CONFUSION MATRIX
# ------------------------------

labels = VALID_CLASSES
cm = confusion_matrix(y_test, y_pred, labels=labels)

plt.figure(figsize=(7, 5))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=labels,
    yticklabels=labels
)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=300)
plt.close()

# ------------------------------
# 6. FEATURE IMPORTANCE
# ------------------------------

feat_imp = pd.Series(
    model.feature_importances_,
    index=X.columns
).sort_values(ascending=False).head(25)

print("\nTop 25 Features:\n")
print(feat_imp)

feat_imp.to_csv("feature_importance.csv")

# Plot
plt.figure(figsize=(10, 6))
feat_imp[::-1].plot(kind="barh")
plt.title("Top 25 Most Important Features")
plt.xlabel("Importance")
plt.tight_layout()
plt.savefig("feature_importance.png", dpi=300)
plt.close()
