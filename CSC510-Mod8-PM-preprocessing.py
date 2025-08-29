import pandas as pd

# Load the original dataset
df = pd.read_csv("digital_marketing_campaign_dataset.csv")

# Define core features (must-have columns)
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

# Keep only the core features (drop everything else)
core_dataframe = df[core_features]

# Save as a new CSV (clone with trimmed fields)
core_dataframe.to_csv("preprocessed_portfolio_milestone.csv", index=False)

print("Trimmed dataset saved as preprocessed_portfolio_milestone.csv")
print("Shape of trimmed dataset:", core_dataframe.shape)
print(core_dataframe.head())
