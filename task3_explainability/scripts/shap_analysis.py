# task3_explainability/scripts/shap_analysis.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import shap
import warnings
warnings.filterwarnings('ignore')
import os

print("=" * 70)
print("TASK 3: SIMPLIFIED SHAP ANALYSIS")
print("=" * 70)

def main():
    """Main function - simplified for easy execution"""
    
    # Step 1: Check if model exists
    model_path = "models/fraud_detection_model.joblib"
    if not os.path.exists(model_path):
        print(f"ERROR: Model not found at {model_path}")
        print("\nFirst train a model:")
        print("Run: python train_simple_model.py")
        return
    
    # Step 2: Check if data exists
    data_path = "data/processed/fraud_processed.csv"
    if not os.path.exists(data_path):
        print(f"ERROR: Data not found at {data_path}")
        print("\nFirst preprocess your data:")
        print("1. Place Fraud_Data.csv in data/raw/")
        print("2. Run: python scripts/preprocess_data.py")
        return
    
    print("✓ Model found:", model_path)
    print("✓ Data found:", data_path)
    
    # Step 3: Load model and data
    print("\n1. Loading model and data...")
    try:
        model = joblib.load(model_path)
        data = pd.read_csv(data_path)
        print(f"   Model type: {type(model).__name__}")
        print(f"   Data shape: {data.shape}")
    except Exception as e:
        print(f"   ERROR loading: {e}")
        return
    
    # Step 4: Prepare features
    print("\n2. Preparing features...")
    
    # Find target column
    target_col = None
    for col in ['class', 'Class', 'is_fraud']:
        if col in data.columns:
            target_col = col
            break
    
    if target_col is None:
        print("   ERROR: Could not find target column!")
        print(f"   Available columns: {list(data.columns)}")
        return
    
    print(f"   Target column: {target_col}")
    
    X = data.drop(columns=[target_col])
    y = data[target_col]
    
    # Keep only numeric columns
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    X = X[numeric_cols]
    feature_names = X.columns.tolist()
    
    print(f"   Using {len(feature_names)} features")
    
    # Step 5: Create SHAP explainer
    print("\n3. Creating SHAP explainer...")
    
    # Use small sample for speed
    if len(X) > 500:
        X_sample = X.sample(500, random_state=42)
        y_sample = y.loc[X_sample.index]
    else:
        X_sample = X
        y_sample = y
    
    # Create explainer based on model type
    model_type = type(model).__name__
    
    try:
        if model_type in ['RandomForestClassifier', 'RandomForestRegressor']:
            explainer = shap.TreeExplainer(model)
            print(f"   Using TreeExplainer for {model_type}")
        else:
            # Use Kernel SHAP for other models
            background = shap.sample(X_sample, min(50, len(X_sample)))
            explainer = shap.KernelExplainer(model.predict_proba, background)
            print(f"   Using KernelExplainer for {model_type}")
    except Exception as e:
        print(f"   ERROR creating explainer: {e}")
        return
    
    # Step 6: Calculate SHAP values
    print("\n4. Calculating SHAP values...")
    try:
        shap_values = explainer.shap_values(X_sample)
        
        # Handle different output formats
        if isinstance(shap_values, list):
            if len(shap_values) == 2:  # Binary classification
                shap_values = shap_values[1]  # Use positive class
            else:
                shap_values = shap_values[0]
        
        print(f"   SHAP values shape: {shap_values.shape}")
    except Exception as e:
        print(f"   ERROR calculating SHAP values: {e}")
        return
    
    # Step 7: Create SHAP summary plot
    print("\n5. Creating SHAP summary plot...")
    plt.figure(figsize=(12, 8))
    
    try:
        shap.summary_plot(shap_values, X_sample, 
                         feature_names=feature_names,
                         max_display=15,
                         show=False)
        
        plt.title("SHAP Feature Importance Summary", fontsize=16, pad=20)
        plt.tight_layout()
        
        # Save plot
        save_path = "task3_explainability/results/shap_summary.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"   ✓ Saved: {save_path}")
    except Exception as e:
        print(f"   ERROR creating summary plot: {e}")
    
    # Step 8: Feature importance analysis
    print("\n6. Analyzing feature importance...")
    try:
        # Calculate SHAP importance
        shap_importance = np.abs(shap_values).mean(axis=0)
        
        # Create importance dataframe
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'shap_importance': shap_importance
        }).sort_values('shap_importance', ascending=False)
        
        # Save to CSV
        importance_df.to_csv("task3_explainability/results/feature_importance.csv", index=False)
        
        # Display top features
        print("\n   Top 10 Most Important Features:")
        print("   " + "-" * 40)
        for idx, row in importance_df.head(10).iterrows():
            print(f"   {row['feature']}: {row['shap_importance']:.4f}")
        
        # Create feature importance plot
        plt.figure(figsize=(10, 6))
        top_features = importance_df.head(10)
        
        plt.barh(range(len(top_features)), top_features['shap_importance'])
        plt.yticks(range(len(top_features)), top_features['feature'])
        plt.xlabel('SHAP Importance')
        plt.title('Top 10 Feature Importances', fontsize=14)
        plt.gca().invert_yaxis()
        plt.tight_layout()
        
        save_path = "task3_explainability/results/feature_importance_plot.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"   ✓ Saved: {save_path}")
        
    except Exception as e:
        print(f"   ERROR analyzing feature importance: {e}")
    
    # Step 9: Business recommendations
    print("\n7. Generating business recommendations...")
    try:
        if 'importance_df' in locals():
            top_features = importance_df.head(5)['feature'].tolist()
            
            recommendations = [
                f"1. Monitor transactions with unusual patterns in: {', '.join(top_features[:3])}",
                "2. Implement additional verification for high-risk transactions",
                "3. Use SHAP values for real-time fraud risk scoring",
                "4. Create alerts for transactions with high SHAP values",
                "5. Regularly update fraud detection rules based on model insights"
            ]
            
            # Save recommendations
            with open("task3_explainability/results/business_recommendations.txt", "w") as f:
                f.write("BUSINESS RECOMMENDATIONS\n")
                f.write("=" * 50 + "\n\n")
                f.write("Based on SHAP analysis of fraud detection model\n\n")
                f.write("Top 5 Fraud Predictors:\n")
                for i, feature in enumerate(top_features, 1):
                    f.write(f"{i}. {feature}\n")
                f.write("\n" + "=" * 50 + "\n\n")
                f.write("Recommendations:\n\n")
                for rec in recommendations:
                    f.write(f"{rec}\n")
            
            print("\n   Top 5 Fraud Predictors:")
            for i, feature in enumerate(top_features, 1):
                print(f"   {i}. {feature}")
            
            print("\n   Key Recommendations:")
            for i, rec in enumerate(recommendations[:3], 1):
                print(f"   {i}. {rec}")
                
            print(f"\n   ✓ Recommendations saved to: task3_explainability/results/business_recommendations.txt")
            
    except Exception as e:
        print(f"   ERROR generating recommendations: {e}")
    
    print("\n" + "=" * 70)
    print("TASK 3 COMPLETE!")
    print("=" * 70)
    print("\nResults saved to: task3_explainability/results/")
    print("\nTo view your results:")
    print("1. Open task3_explainability/results/shap_summary.png")
    print("2. Check task3_explainability/results/business_recommendations.txt")

if __name__ == "__main__":
    main()