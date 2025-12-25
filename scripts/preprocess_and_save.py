import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# -----------------------------
# FRAUD DATA PROCESSING
# -----------------------------

fraud = pd.read_csv("../data/raw/Fraud_Data.csv")
ip = pd.read_csv("../data/raw/IpAddress_to_Country.csv")

# Convert times
fraud["signup_time"] = pd.to_datetime(fraud["signup_time"])
fraud["purchase_time"] = pd.to_datetime(fraud["purchase_time"])

# Feature engineering
fraud["hour_of_day"] = fraud["purchase_time"].dt.hour
fraud["day_of_week"] = fraud["purchase_time"].dt.dayofweek
fraud["time_since_signup"] = (
    fraud["purchase_time"] - fraud["signup_time"]
).dt.total_seconds() / 3600

# Convert IP to integer (already numeric in this dataset)
fraud["ip_int"] = fraud["ip_address"].astype(np.int64)

# Prepare IP mapping
ip["lower_bound_ip_address"] = ip["lower_bound_ip_address"].astype(np.int64)
ip["upper_bound_ip_address"] = ip["upper_bound_ip_address"].astype(np.int64)

ip = ip.sort_values("lower_bound_ip_address")
fraud = fraud.sort_values("ip_int")

# Merge country
fraud = pd.merge_asof(
    fraud,
    ip,
    left_on="ip_int",
    right_on="lower_bound_ip_address",
    direction="backward"
)

# Select final features
fraud_final = fraud[
    [
        "purchase_value",
        "hour_of_day",
        "day_of_week",
        "time_since_signup",
        "class"
    ]
]

# Scale numeric features
scaler = StandardScaler()
num_cols = fraud_final.drop(columns="class").columns
fraud_final[num_cols] = scaler.fit_transform(fraud_final[num_cols])

# Save
fraud_final.to_csv("../data/processed/fraud_processed.csv", index=False)


# -----------------------------
# CREDIT CARD DATA PROCESSING
# -----------------------------

credit = pd.read_csv("../data/raw/creditcard.csv")

X = credit.drop("Class", axis=1)
y = credit["Class"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

credit_final = pd.DataFrame(X_scaled, columns=X.columns)
credit_final["Class"] = y.values

# Save
credit_final.to_csv("../data/processed/creditcard_processed.csv", index=False)

print("✅ Processed files created successfully")
