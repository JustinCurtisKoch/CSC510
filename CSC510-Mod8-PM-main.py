
# AI-Powered Marketing Channel Attribution Tool

# ------------------------------
# Imports
# ------------------------------
import os
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
    auc
)
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

# ------------------------------
# 1. Load cleaned dataset
# ------------------------------
df = pd.read_csv("cleaned_portfolio_milestone.csv")
os.makedirs("figures", exist_ok=True)  # Ensure output directory exists

# ------------------------------
# 2. Exploratory Data Analysis (EDA)
# ------------------------------

# Define numeric columns for visualization
numeric_cols = [
    "AdSpend", "ClickThroughRate", "EmailOpens",
    "EmailClicks", "PreviousPurchases", "WebsiteVisits"
]

# Distribution plots for numeric features
for col in numeric_cols:
    plt.figure(figsize=(6, 4))
    sns.histplot(df[col], kde=True, bins=30)
    plt.title(f"Distribution of {col}")
    plt.savefig(f"figures/distribution_{col}.png")
    plt.close()

# Class balance for target variable
plt.figure(figsize=(5, 4))
sns.countplot(x="Conversion", data=df)
plt.title("Target Variable Distribution (Conversion)")
plt.savefig("figures/conversion_balance.png")
plt.close()

# Correlation heatmap
plt.figure(figsize=(10, 8))
corr = df.corr()
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap")
plt.savefig("figures/correlation_heatmap.png")
plt.close()

# Boxplots for features vs. target variable
for col in numeric_cols:
    plt.figure(figsize=(6, 4))
    sns.boxplot(x="Conversion", y=col, data=df)
    plt.title(f"{col} by Conversion")
    plt.savefig(f"figures/{col}_by_conversion.png")
    plt.close()

# ------------------------------
# 3. Train/Test Split
# ------------------------------
X = df.drop(columns=["Conversion"])  # Features
y = df["Conversion"]                 # Target variable

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ------------------------------
# 4. Baseline Logistic Regression
# ------------------------------
log_reg = LogisticRegression(max_iter=1000, random_state=42)
log_reg.fit(X_train, y_train)

y_pred_lr = log_reg.predict(X_test)
y_proba_lr = log_reg.predict_proba(X_test)[:, 1]

# ------------------------------
# 5. Logistic Regression with SMOTE
# ------------------------------
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

log_reg_smote = LogisticRegression(max_iter=1000, solver="liblinear")
log_reg_smote.fit(X_train_res, y_train_res)

y_pred_lr_smote = log_reg_smote.predict(X_test)
y_pred_proba_smote = log_reg_smote.predict_proba(X_test)[:, 1]

# ------------------------------
# 6. ROC and Precision-Recall Curves
# ------------------------------
fpr, tpr, _ = roc_curve(y_test, y_proba_lr)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6, 6))
plt.plot(fpr, tpr, color="blue", lw=2, label=f"ROC curve (AUC = {roc_auc:.2f})")
plt.plot([0, 1], [0, 1], color="gray", linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate (Recall)")
plt.title("Receiver Operating Characteristic (ROC)")
plt.legend(loc="lower right")
plt.savefig("figures/roc_curve.png")
plt.close()

precision, recall, _ = precision_recall_curve(y_test, y_proba_lr)
plt.figure(figsize=(6, 6))
plt.plot(recall, precision, color="green", lw=2)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.savefig("figures/precision_recall_curve.png")
plt.close()

# ------------------------------
# 7. Random Forest Classifier
# ------------------------------
rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=None,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)
rf_model.fit(X_train, y_train)

rf_y_pred = rf_model.predict(X_test)
rf_y_proba = rf_model.predict_proba(X_test)[:, 1]

# ------------------------------
# 8. Gradient Boosted Trees (XGBoost)
# ------------------------------
xgb_model = XGBClassifier(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=(y_train.value_counts()[0] / y_train.value_counts()[1]),
    random_state=42,
    use_label_encoder=False,
    eval_metric="logloss"
)
xgb_model.fit(X_train, y_train)

xgb_y_pred = xgb_model.predict(X_test)
xgb_y_proba = xgb_model.predict_proba(X_test)[:, 1]

# Feature importances from Gradient Boosted Trees
feature_importances = xgb_model.feature_importances_
importance_df = pd.DataFrame({
    "Feature": X.columns,
    "Importance": feature_importances
}).sort_values(by="Importance", ascending=False)

plt.figure(figsize=(8, 6))
sns.barplot(x="Importance", y="Feature", data=importance_df, palette="viridis")
plt.title("Feature Importance - Gradient Boosted Trees")
plt.tight_layout()
plt.savefig("figures/gbt_feature_importance.png")
plt.close()

# ------------------------------
# 9. Results Summary (Consolidated Output)
# ------------------------------
print("\n========== MODEL PERFORMANCE SUMMARY ==========\n")

print("=== Logistic Regression (Baseline) ===")
print(f"Accuracy: {accuracy_score(y_test, y_pred_lr):.4f}")
print(f"ROC AUC: {roc_auc_score(y_test, y_proba_lr):.4f}")
print("Classification Report:\n", classification_report(y_test, y_pred_lr))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_lr))

print("\n=== Logistic Regression with SMOTE ===")
print(f"Accuracy: {accuracy_score(y_test, y_pred_lr_smote):.4f}")
print(f"ROC AUC: {roc_auc_score(y_test, y_pred_proba_smote):.4f}")
print("Classification Report:\n", classification_report(y_test, y_pred_lr_smote))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_lr_smote))

print("\n=== Random Forest Classifier ===")
print(f"Accuracy: {accuracy_score(y_test, rf_y_pred):.4f}")
print(f"ROC AUC: {roc_auc_score(y_test, rf_y_proba):.4f}")
print("Classification Report:\n", classification_report(y_test, rf_y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, rf_y_pred))

print("\n=== Gradient Boosted Trees ===")
print(f"Accuracy: {accuracy_score(y_test, xgb_y_pred):.4f}")
print(f"ROC AUC: {roc_auc_score(y_test, xgb_y_proba):.4f}")
print("Classification Report:\n", classification_report(y_test, xgb_y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, xgb_y_pred))

print("\n=== Feature Importances (Gradient Boosted Trees) ===")
print(importance_df)

print("\n========== END OF SUMMARY ==========")
