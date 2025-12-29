# test_shap.py
print("Testing SHAP installation and basic functionality...")
print("=" * 60)

try:
    # Try to import SHAP
    import shap
    print(f"✓ SHAP imported successfully!")
    print(f"  Version: {shap.__version__}")
except ImportError as e:
    print(f"✗ SHAP import failed: {e}")
    print("\nTo install SHAP, run:")
    print("  pip install shap")
    exit()

try:
    # Try to import other required libraries
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import sklearn
    print(f"✓ Pandas: {pd.__version__}")
    print(f"✓ NumPy: {np.__version__}")
    print(f"✓ Scikit-learn: {sklearn.__version__}")
except ImportError as e:
    print(f"✗ Import failed: {e}")

# Create simple test data
print("\nCreating test data...")
np.random.seed(42)

# Create synthetic data
n_samples = 100
X = pd.DataFrame({
    'feature1': np.random.randn(n_samples),
    'feature2': np.random.randn(n_samples),
    'feature3': np.random.randn(n_samples)
})
y = (X['feature1'] + 2*X['feature2'] + np.random.randn(n_samples) > 0).astype(int)

print(f"Test data created: {X.shape}")
print(f"Target distribution: {pd.Series(y).value_counts().to_dict()}")

# Train a simple model
print("\nTraining simple model...")
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=10, random_state=42)
model.fit(X, y)

print(f"Model trained: {type(model).__name__}")

# Test SHAP functionality
print("\nTesting SHAP functionality...")
try:
    # Create SHAP explainer
    explainer = shap.TreeExplainer(model)
    print("✓ TreeExplainer created")
    
    # Calculate SHAP values
    shap_values = explainer.shap_values(X)
    print(f"✓ SHAP values calculated: {np.array(shap_values).shape}")
    
    # Create simple plot
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X, show=False)
    plt.title("SHAP Test Plot", fontsize=14)
    plt.tight_layout()
    
    # Save plot
    import os
    os.makedirs("task3_explainability/results", exist_ok=True)
    plt.savefig("task3_explainability/results/test_shap_plot.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✓ Test plot created and saved")
    print(f"  Saved to: task3_explainability/results/test_shap_plot.png")
    
except Exception as e:
    print(f"✗ SHAP test failed: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
print("TEST COMPLETE!")
print("=" * 60)

print("\nIf everything worked, you should see:")
print("✓ SHAP imported")
print("✓ Test data created")
print("✓ Model trained")
print("✓ SHAP explainer created")
print("✓ Plot saved to task3_explainability/results/")