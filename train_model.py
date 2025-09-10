import pandas as pd
import joblib
import json
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture

# Load internal dataset
df = pd.read_csv("audience_data.csv")

# Features to train on
features = [
    "Total_Spend", "Purchase_Frequency", "Avg_Order_Value",
    "Last_Purchase_Days_Ago", "Loyalty_Score",
    "Impressions", "Clicks", "Spent", "Total_Conversion", "Approved_Conversion"
]

# Preprocess
df = df.fillna(0)
X = df[features].values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train GMM
gmm = GaussianMixture(n_components=3, covariance_type='full', random_state=42)
gmm.fit(X_scaled)

# Save artifacts
joblib.dump(gmm, "gmm_model.pkl")
joblib.dump(scaler, "scaler.pkl")
with open("feature_list.json", "w") as f:
    json.dump(features, f)

print("Model trained & saved: gmm_model.pkl, scaler.pkl, feature_list.json")
