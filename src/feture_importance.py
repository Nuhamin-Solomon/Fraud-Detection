import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.inspection import permutation_importance
import warnings
warnings.filterwarnings('ignore')
import os

print("=" * 60)
print("FEATURE IMPORTANCE ANALYSIS")
print("=" * 60)

def analyze_feature_importance(model_path, data_path, results_dir="task3_explainability/results"):
    """Comprehensive feature importance analysis"""
    
    # Create results directory
    os.makedirs(results_dir, exist_ok=True)
    
    # Load model and data
    print("\n1. Loading model and data...")
    model = joblib.load(model_path)
    data = pd.read_csv(data_path)
    
    # Prepare features and target
    X = data.drop('class', axis=1)
    y = data['class']
    feature_names = X.columns.tolist()
    
    print(f"✓ Model: {type(model).__name__}")
    print(f"✓ Features: {len(feature_names)}")
    print(f"✓ Samples: {len(X)}")
    
    # 1. Built-in feature importance
    print("\n2. Analyzing built-in feature importance...")
    
    if hasattr(model, 'feature_importances_'):
        importance = model.feature_importances_
        
        # Create importance dataframe
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        print("\nTop 10 features (built-in):")
        for idx, row in importance_df.head(10).iterrows():
            print(f"  {row['feature']}: {row['importance']:.4f}")
        
        # Plot
        plt.figure(figsize=(12, 8))
        top_features = importance_df.head(15)
        plt.barh(range(len(top_features)), top_features['importance'])
        plt.yticks(range(len(top_features)), top_features['feature'])
        plt.xlabel('Importance Score')
        plt.title('Built-in Feature Importance (Top 15)', fontsize=16)
        plt.gca().invert_yaxis()
        plt.tight_layout()
        
        save_path = os.path.join(results_dir, "builtin_feature_importance.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"✓ Plot saved: {save_path}")
    
    # 2. Permutation importance
    print("\n3. Calculating permutation importance...")
    
    try:
        # Use smaller sample for speed
        if len(X) > 1000:
            X_sample = X.sample(1000, random_state=42)
            y_sample = y.loc[X_sample.index]
        else:
            X_sample = X
            y_sample = y
        
        # Calculate permutation importance
        perm_importance = permutation_importance(
            model, X_sample, y_sample,
            n_repeats=10,
            random_state=42,
            n_jobs=-1
        )
        
        # Create dataframe
        perm_df = pd.DataFrame({
            'feature': feature_names,
            'importance_mean': perm_importance.importances_mean,
            'importance_std': perm_importance.importances_std
        }).sort_values('importance_mean', ascending=False)
        
        print("\nTop 10 features (permutation):")
        for idx, row in perm_df.head(10).iterrows():
            print(f"  {row['feature']}: {row['importance_mean']:.4f} (±{row['importance_std']:.4f})")
        
        # Plot with error bars
        plt.figure(figsize=(12, 8))
        top_perm = perm_df.head(15)
        x_pos = np.arange(len(top_perm))
        
        plt.barh(x_pos, top_perm['importance_mean'], xerr=top_perm['importance_std'],
                capsize=5, alpha=0.7)
        plt.yticks(x_pos, top_perm['feature'])
        plt.xlabel('Decrease in Accuracy')
        plt.title('Permutation Importance with Error Bars (Top 15)', fontsize=16)
        plt.gca().invert_yaxis()
        plt.tight_layout()
        
        save_path = os.path.join(results_dir, "permutation_importance.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"✓ Plot saved: {save_path}")
        
        # Save results
        perm_df.to_csv(os.path.join(results_dir, "permutation_importance.csv"), index=False)
        
    except Exception as e:
        print(f"⚠ Could not calculate permutation importance: {e}")
    
    # 3. Feature correlations with target
    print("\n4. Analyzing feature correlations...")
    
    # Calculate correlations
    corr_with_target = X.corrwith(y)
    corr_df = pd.DataFrame({
        'feature': corr_with_target.index,
        'correlation': corr_with_target.values
    }).sort_values('correlation', key=abs, ascending=False)
    
    print("\nTop 10 correlated features:")
    for idx, row in corr_df.head(10).iterrows():
        print(f"  {row['feature']}: {row['correlation']:.4f}")
    
    # Plot correlation
    plt.figure(figsize=(12, 8))
    top_corr = corr_df.head(15)
    colors = ['red' if x < 0 else 'blue' for x in top_corr['correlation']]
    
    plt.barh(range(len(top_corr)), top_corr['correlation'], color=colors)
    plt.yticks(range(len(top_corr)), top_corr['feature'])
    plt.xlabel('Correlation with Target')
    plt.title('Feature Correlation with Fraud (Top 15)', fontsize=16)
    plt.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
    plt.gca().invert_yaxis()
    plt.tight_layout()
    
    save_path = os.path.join(results_dir, "feature_correlations.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"✓ Plot saved: {save_path}")
    
    # Save all results
    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE!")
    print("=" * 60)
    print(f"\nResults saved to: {results_dir}")
    print("\nFiles created:")
    print("✓ builtin_feature_importance.png")
    print("✓ permutation_importance.png")
    print("✓ feature_correlations.png")
    print("✓ permutation_importance.csv")

if __name__ == "__main__":
    # Update these paths
    model_path = "models/fraud_detection_model.joblib"
    data_path = "data/processed/fraud_processed.csv"
    
    analyze_feature_importance(model_path, data_path)