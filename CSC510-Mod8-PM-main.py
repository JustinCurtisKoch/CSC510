import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ------------------------------
# Load cleaned dataset
# ------------------------------
df = pd.read_csv("cleaned_portfolio_milestone.csv")

# Make sure output directory exists
os.makedirs("figures", exist_ok=True)

print("=== Dataset Shape ===")
print(df.shape)
print("\n=== Dataset Head ===")
print(df.head())

# ------------------------------
# 1. Distribution of numeric features
# ------------------------------
numeric_cols = ['AdSpend', 'ClickThroughRate', 'EmailOpens', 
                'EmailClicks', 'PreviousPurchases', 'WebsiteVisits']

for col in numeric_cols:
    plt.figure(figsize=(6,4))
    sns.histplot(df[col], kde=True, bins=30)
    plt.title(f"Distribution of {col}")
    plt.savefig(f"figures/distribution_{col}.png")
    plt.close()

# ------------------------------
# 2. Class balance for Conversion
# ------------------------------
plt.figure(figsize=(5,4))
sns.countplot(x="Conversion", data=df)
plt.title("Target Variable Distribution (Conversion)")
plt.savefig("figures/conversion_balance.png")
plt.close()

# ------------------------------
# 3. Correlation heatmap
# ------------------------------
plt.figure(figsize=(10,8))
corr = df.corr()
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap")
plt.savefig("figures/correlation_heatmap.png")
plt.close()

# ------------------------------
# 4. Feature vs. Target boxplots
# ------------------------------
for col in numeric_cols:
    plt.figure(figsize=(6,4))
    sns.boxplot(x="Conversion", y=col, data=df)
    plt.title(f"{col} by Conversion")
    plt.savefig(f"figures/{col}_by_conversion.png")
    plt.close()

print("EDA complete. Figures saved in /figures/")

# ------------------------------
# 5. Train/Test Split
# ------------------------------
X = df.drop(columns=["Conversion"])
y = df["Conversion"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("\n=== Train/Test Split ===")
print(f"Training set size: {X_train.shape}")
print(f"Test set size: {X_test.shape}")

# ------------------------------
# 6. Baseline Logistic Regression
# ------------------------------
log_reg = LogisticRegression(max_iter=1000, random_state=42)
log_reg.fit(X_train, y_train)

y_pred = log_reg.predict(X_test)

print("\n=== Logistic Regression Performance ===")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("Classification Report:")
print(classification_report(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# ------------------------------
# 7. Model Evaluation (placeholder)
# ------------------------------
# TODO: Add ROC curve, precision-recall curve, AUC metrics

# ------------------------------
# 8. Next Steps (placeholder)
# ------------------------------
# TODO: Add more advanced models (Random Forest, XGBoost, etc.)
# TODO: Hyperparameter tuning
