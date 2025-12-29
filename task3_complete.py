# task3_minimal.py
print("TASK 3 - MINIMAL VERSION")
print("=" * 50)

# Install if needed
import subprocess
import sys
try:
    import shap
except:
    print("Installing SHAP...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "shap", "-q"])

# Import
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap

# Create simple data
np.random.seed(42)
data = pd.DataFrame({
    'amount': np.random.exponential(100, 100),
    'time': np.random.randint(0, 24, 100),
    'frequency': np.random.poisson(5, 100),
    'fraud': (np.random.rand(100) > 0.9).astype(int)
})

print(f"Data: {data.shape}")
print(f"Fraud: {data['fraud'].sum()} cases")

# Train model
from sklearn.ensemble import RandomForestClassifier
X = data.drop('fraud', axis=1)
y = data['fraud']

model = RandomForestClassifier(n_estimators=50, random_state=42)
model.fit(X, y)

# SHAP
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)

if isinstance(shap_values, list):
    shap_values = shap_values[1]

# Plot
plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values, X, show=False)
plt.title("SHAP Analysis - Fraud Detection", fontsize=14)
plt.tight_layout()

# Save
import os
os.makedirs("task3_explainability/results", exist_ok=True)
plt.savefig("task3_explainability/results/minimal_shap.png", dpi=300)
plt.show()

print("\n✓ Plot saved: task3_explainability/results/minimal_shap.png")

# Recommendations
with open("task3_explainability/results/minimal_recommendations.txt", "w") as f:
    f.write("MINIMAL FRAUD DETECTION RECOMMENDATIONS\n")
    f.write("=" * 40 + "\n\n")
    f.write("1. Monitor transaction amounts\n")
    f.write("2. Check transaction timing\n")
    f.write("3. Watch for unusual frequency\n")
    f.write("4. Use SHAP for risk scoring\n")

print("✓ Recommendations saved")

print("\n" + "=" * 50)
print("TASK 3 COMPLETE!")
print("=" * 50)