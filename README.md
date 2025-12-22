# Fraud-Detection
Creating and engineering features that help identify fraud patterns.
Dataset Description
1. Fraud_Data.csv

Columns: user_id, signup_time, purchase_time, purchase_value, ip_address, class

Target Variable: class (1 = fraud, 0 = legitimate)

2. IpAddress_to_Country.csv

Mapping of IP address ranges to countries

Columns: lower_bound_ip_address, upper_bound_ip_address, country

3. creditcard.csv

Credit card transactions with anonymized features (V1–V28) and transaction Amount

Target Variable: Class (1 = fraud, 0 = legitimate)

Task 1: Data Preprocessing & EDA
Step 1: Data Cleaning

Removed missing values from all datasets.

Dropped duplicate rows to ensure dataset integrity.

Converted datetime columns (signup_time, purchase_time) to proper datetime types.

Step 2: Exploratory Data Analysis (EDA)

Class Distribution:

Fraud_Data.csv: ~1–2% fraudulent transactions

creditcard.csv: ~0.17% fraudulent transactions

Univariate Analysis:

Distribution of purchase_value in Fraud_Data.csv

Distribution of Amount in creditcard.csv

Bivariate Analysis:

Boxplots of purchase_value vs class to observe fraud patterns

Visualizations produced for both datasets using matplotlib and seaborn.

Step 3: Geolocation Integration

Converted ip_address to integer for Fraud_Data.csv.

Merged Fraud_Data with IpAddress_to_Country.csv using range-based merge_asof.

Analyzed fraud patterns by country:

Top countries with highest fraud rate reported.

Step 4: Feature Engineering

Time-based features:

hour_of_day — hour of purchase

day_of_week — day of purchase

time_since_signup — hours between signup and purchase

Transaction frequency:

txn_count_24h — number of transactions per user in 24h windows

Focused on numeric/time features to reduce memory overhead.

Step 5: Data Transformation

Standardized numeric features using StandardScaler.

Avoided high-cardinality one-hot encoding to prevent memory errors.

Step 6: Handling Class Imbalance

Train-test split with stratification to preserve class ratios.

Applied SMOTE to training set only to balance minority class.

Documented class distribution before and after SMOTE.

Key Insights from Task 1

Both datasets are highly imbalanced; fraud occurs in <2% of transactions.

Fraud rates differ significantly by country in Fraud_Data.csv.

Time-based features like hour_of_day and time_since_signup provide strong signals for detecting fraud.

SMOTE effectively balances the training data while preserving patterns for model training.

Files and Structure
fraud-detection/
├── data/
│   ├── raw/               # Original datasets (Fraud_Data.csv, creditcard.csv)
│   └── processed/         # Processed datasets after Task 1
├── notebooks/
│   ├── eda-fraud-data.ipynb
│   ├── eda-creditcard.ipynb
│   └── feature-engineering.ipynb
├── src/                   # Optional scripts for preprocessing
├── models/                # Placeholder for trained models
├── requirements.txt
├── README.md
└── .gitignore

Next Steps

Task 2: Build and evaluate baseline and ensemble models for fraud detection.

Task 3: Apply SHAP explainability to understand feature importance and business insights.
