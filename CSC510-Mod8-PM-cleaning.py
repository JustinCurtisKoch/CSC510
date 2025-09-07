# CSC510 Foundations of Artificial Intelligence
# Module 8: Portfolio Milestone
# Data Cleaning Script

# ------------------------------
# Imports
# ------------------------------
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# ------------------------------
# 1. Load Preprocessed Dataset
# ------------------------------
df = pd.read_csv("preprocessed_portfolio_milestone.csv")

# ------------------------------
# 2. Encode Categorical Variables
# ------------------------------
categorical_cols = ["CampaignChannel", "CampaignType"]

# One-hot encode categories (drop first to avoid dummy variable trap)
encoder = OneHotEncoder(sparse_output=False, drop="first")
encoded = encoder.fit_transform(df[categorical_cols])

# Create new DataFrame for encoded features with names
encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(categorical_cols))

# Drop original categorical columns and add encoded versions
df = df.drop(columns=categorical_cols).join(encoded_df)

# ------------------------------
# 3. Scale Continuous Features
# ------------------------------
continuous_cols = [
    "AdSpend",
    "ClickThroughRate",
    "EmailOpens",
    "EmailClicks",
    "PreviousPurchases",
    "WebsiteVisits"
]

# Apply standard scaling
scaler = StandardScaler()
df[continuous_cols] = scaler.fit_transform(df[continuous_cols])

# ------------------------------
# 4. Save Cleaned Dataset
# ------------------------------
df.to_csv("cleaned_portfolio_milestone.csv", index=False)

# ------------------------------
# 5. Output Summary
# ------------------------------
print("\n========== CLEANING SUMMARY ==========\n")
print("Cleaned dataset saved as: cleaned_portfolio_milestone.csv")
print(f"Final dataset shape: {df.shape}")
print("\nFinal columns:\n", df.columns.tolist())
print("\n========== END OF SUMMARY ==========")

