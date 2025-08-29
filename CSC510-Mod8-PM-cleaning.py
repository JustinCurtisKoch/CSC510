import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# === Load dataset ===
df = pd.read_csv("preprocessed_portfolio_milestone.csv")

# Continuous variables
continuous_vars = ["AdSpend", "ClickThroughRate", "EmailOpens", "EmailClicks", "WebsiteVisits"]

# Compute mean and standard deviation
stats = df[continuous_vars].agg(['mean', 'std']).T
print("=== Continuous Variable Stats (Before Scaling) ===")
print(stats)

# === 1. Encode categorical variables ===
categorical_cols = ["CampaignChannel", "CampaignType"]
encoder = OneHotEncoder(sparse_output=False, drop="first")  # drop first to avoid dummy variable trap
encoded = encoder.fit_transform(df[categorical_cols])

# Create encoded dataframe with proper column names
encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(categorical_cols))

# Drop original categorical columns and join encoded
df = df.drop(columns=categorical_cols).join(encoded_df)

# === 2. Scale continuous features ===
continuous_cols = ["AdSpend", "ClickThroughRate", "EmailOpens", "EmailClicks", "PreviousPurchases", "WebsiteVisits"]

scaler = StandardScaler()
df[continuous_cols] = scaler.fit_transform(df[continuous_cols])

# === 3. Review correlations (optional step here) ===
correlations = df.corr()

# === 4. Save cleaned dataset ===
df.to_csv("cleaned_portfolio_milestone.csv", index=False)

print("Cleaning complete. Saved as cleaned_portfolio_milestone.csv")
print("Final columns:", df.columns.tolist())
