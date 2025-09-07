# CSC510 Foundations of Artificial Intelligence
# Module 8: Portfolio Milestone
# Data Preprocessing Script

# ------------------------------
# Imports
# ------------------------------
import pandas as pd

# ------------------------------
# 1. Load the Original Dataset
# ------------------------------
df = pd.read_csv("digital_marketing_campaign_dataset.csv")

# ------------------------------
# 2. Select Core Features
# ------------------------------
# Define essential features to retain
core_features = [
    "CampaignChannel",
    "CampaignType",
    "AdSpend",
    "ClickThroughRate",
    "EmailOpens",
    "EmailClicks",
    "PreviousPurchases",
    "WebsiteVisits",
    "Conversion"
]

# Keep only the selected features
core_df = df[core_features]

# ------------------------------
# 3. Save Trimmed Dataset
# ------------------------------
core_df.to_csv("preprocessed_portfolio_milestone.csv", index=False)

# ------------------------------
# 4. Output Summary
# ------------------------------
print("\n========== PREPROCESSING SUMMARY ==========\n")
print("Trimmed dataset saved as: preprocessed_portfolio_milestone.csv")
print(f"Shape of trimmed dataset: {core_df.shape}")
print("\nPreview of data:\n")
print(core_df.head())
print("\n========== END OF SUMMARY ==========")
