"""
Wine Quality Model Interpretation

This script performs advanced model interpretation to extract insights from trained models.
It includes:
- Partial dependence plots to show how individual features affect quality
- Threshold analysis to identify critical values for key features
- Domain-specific insights based on wine chemistry

Usage:
    python src/models/model_interpretation.py [--wine-type {red,white,combined}]

Author: Your Name
Date: March 2025
"""

import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from pathlib import Path
from sklearn.inspection import partial_dependence, PartialDependenceDisplay
from sklearn.inspection import permutation_importance
import shap
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import warnings

# Suppress FutureWarnings
warnings.filterwarnings("ignore", category=FutureWarning)

# Define paths
ROOT_DIR = Path(__file__).resolve().parents[2]
ENGINEERED_DATA_DIR = ROOT_DIR / "data" / "processed" / "engineered"
MODELS_DIR = ROOT_DIR / "models"
REPORTS_DIR = ROOT_DIR / "reports"
INTERPRETATION_DIR = REPORTS_DIR / "interpretation"

# Create directories if they don't exist
os.makedirs(INTERPRETATION_DIR, exist_ok=True)

# Define key features for interpretation based on domain knowledge
KEY_FEATURES = {
    'red': [
        'alcohol', 'volatile acidity', 'sulphates', 'total sulfur dioxide',
        'pH', 'density', 'fixed acidity', 'free_to_total_so2_ratio', 'body_score'
    ],
    'white': [
        'alcohol', 'volatile acidity', 'residual sugar', 'total sulfur dioxide',
        'density', 'pH', 'free sulfur dioxide', 'free_to_total_so2_ratio', 'body_score'
    ],
    'combined': [
        'alcohol', 'volatile acidity', 'residual sugar', 'total sulfur dioxide',
        'sulphates', 'pH', 'density', 'free_to_total_so2_ratio', 'body_score'
    ]
}

# Wine chemistry domain knowledge for interpretation
WINE_CHEMISTRY_INSIGHTS = {
    'alcohol': {
        'name': 'Alcohol Content',
        'description': 'Higher alcohol content is typically associated with higher quality wine, contributing to body and warmth.',
        'expected_effect': 'positive',
        'typical_range': {'red': '11-14%', 'white': '10-13%'},
        'critical_threshold': {'red': 12, 'white': 11.5},
        'explanation': 'Alcohol is produced during fermentation. Higher sugar content in grapes leads to higher potential alcohol. More mature grapes tend to produce higher quality wine.'
    },
    'volatile acidity': {
        'name': 'Volatile Acidity',
        'description': 'Measures the gaseous acids in wine, primarily acetic acid. High levels can give a vinegar taste.',
        'expected_effect': 'negative',
        'typical_range': {'red': '0.3-0.7 g/L', 'white': '0.1-0.4 g/L'},
        'critical_threshold': {'red': 0.7, 'white': 0.4},
        'explanation': 'Excessive volatile acidity indicates spoilage by acetic acid bacteria. Lower levels typically mean better wine quality and preservation.'
    },
    'sulphates': {
        'name': 'Sulphates',
        'description': 'Added as a preservative to prevent oxidation and bacterial growth, affecting taste and aroma.',
        'expected_effect': 'positive (to a point)',
        'typical_range': {'red': '0.4-0.9 g/L', 'white': '0.3-0.6 g/L'},
        'critical_threshold': {'red': 0.8, 'white': 0.5},
        'explanation': 'Sulphates help prevent wine spoilage and oxidation. Higher levels (within reasonable limits) tend to correlate with better quality.'
    },
    'total sulfur dioxide': {
        'name': 'Total Sulfur Dioxide',
        'description': 'Sum of free and bound forms of SO2. Helps preserve wine but excessive amounts can cause off-flavors.',
        'expected_effect': 'negative above threshold',
        'typical_range': {'red': '50-150 mg/L', 'white': '100-200 mg/L'},
        'critical_threshold': {'red': 150, 'white': 200},
        'explanation': 'While some SO2 is necessary for preservation, excessive amounts can mask aromas and flavors, and cause a sulfurous taste.'
    },
    'residual sugar': {
        'name': 'Residual Sugar',
        'description': 'Amount of sugar remaining after fermentation, affects sweetness perception.',
        'expected_effect': 'style dependent',
        'typical_range': {'red': '0.5-4 g/L', 'white': '1.5-45 g/L'},
        'critical_threshold': {'red': 2.5, 'white': 6},
        'explanation': 'For dry wines, low residual sugar is preferred. For dessert wines, higher levels are expected. In white wines, sugar balance is crucial for quality.'
    },
    'pH': {
        'name': 'pH',
        'description': 'Measure of acidity. Lower pH (more acidic) improves microbial stability but can increase tartness.',
        'expected_effect': 'optimal range',
        'typical_range': {'red': '3.3-3.7', 'white': '3.0-3.4'},
        'critical_threshold': {'red': 3.6, 'white': 3.3},
        'explanation': 'Lower pH improves wine stability and aging potential. However, extremely low pH can make wine too tart, while high pH makes wine flat and more prone to spoilage.'
    },
    'density': {
        'name': 'Density',
        'description': 'Indicator of alcohol and sugar content. Lower density typically indicates higher alcohol.',
        'expected_effect': 'negative',
        'typical_range': {'red': '0.990-0.997 g/cm³', 'white': '0.987-0.998 g/cm³'},
        'critical_threshold': {'red': 0.995, 'white': 0.992},
        'explanation': 'Density decreases with alcohol content and increases with sugar content. Lower density (higher alcohol) often correlates with higher quality.'
    },
    'fixed acidity': {
        'name': 'Fixed Acidity',
        'description': 'Non-volatile acids (primarily tartaric) that contribute to wine structure and taste.',
        'expected_effect': 'optimal range',
        'typical_range': {'red': '6-9 g/L', 'white': '5-7 g/L'},
        'critical_threshold': {'red': 7.5, 'white': 6.5},
        'explanation': 'Fixed acidity provides structure and aging potential. Too low can make wine flabby, too high can make it too tart.'
    },
    'free sulfur dioxide': {
        'name': 'Free Sulfur Dioxide',
        'description': 'Active form of SO2 that protects wine from oxidation and microbial spoilage.',
        'expected_effect': 'optimal range',
        'typical_range': {'red': '20-40 mg/L', 'white': '30-50 mg/L'},
        'critical_threshold': {'red': 30, 'white': 40},
        'explanation': 'Sufficient free SO2 is essential for wine stability, but excessive levels can cause off-aromas and flavors.'
    },
    'free_to_total_so2_ratio': {
        'name': 'Free to Total SO2 Ratio',
        'description': 'Indicates how much of the sulfur dioxide is in active form to protect the wine.',
        'expected_effect': 'positive',
        'typical_range': {'red': '0.3-0.6', 'white': '0.35-0.65'},
        'critical_threshold': {'red': 0.4, 'white': 0.5},
        'explanation': 'Higher ratios indicate more effective protection against oxidation and spoilage, which is generally associated with better quality.'
    },
    'body_score': {
        'name': 'Body Score (derived)',
        'description': 'Derived feature combining alcohol and density to estimate wine body/mouthfeel.',
        'expected_effect': 'positive',
        'typical_range': {'red': '0.7-1.2', 'white': '0.6-1.0'},
        'critical_threshold': {'red': 0.9, 'white': 0.8},
        'explanation': 'Fuller-bodied wines (higher score) are often perceived as higher quality, particularly in red wines.'
    },
    'citric acid': {
        'name': 'Citric Acid',
        'description': 'Contributes to freshness and fruity flavors in wine.',
        'expected_effect': 'positive (in moderation)',
        'typical_range': {'red': '0-0.5 g/L', 'white': '0-0.5 g/L'},
        'critical_threshold': {'red': 0.3, 'white': 0.35},
        'explanation': 'Citric acid adds fresh, fruity character, but is unstable and can be metabolized by bacteria, potentially leading to spoilage in high amounts.'
    },
    'chlorides': {
        'name': 'Chlorides',
        'description': 'Salt content in wine, affecting taste and perceived minerality.',
        'expected_effect': 'negative above threshold',
        'typical_range': {'red': '0.05-0.1 g/L', 'white': '0.04-0.09 g/L'},
        'critical_threshold': {'red': 0.09, 'white': 0.08},
        'explanation': 'Excessive chlorides can give a salty taste. Moderate levels can contribute to complexity but high levels detract from quality.'
    }
}

def load_data_and_models(wine_type='red'):
    """
    Load engineered data and trained models for interpretation.
    
    Args:
        wine_type: Type of wine ('red', 'white', or 'combined')
        
    Returns:
        tuple: (data, regression_model, binary_model, scaler)
    """
    print(f"\nLoading {wine_type} wine data and models...")
    
    # Load data
    data_path = ENGINEERED_DATA_DIR / f"{wine_type}_wine_engineered.csv"
    data = pd.read_csv(data_path)
    
    # Load models
    regression_model_path = MODELS_DIR / f"{wine_type}_wine_regression_model.pkl"
    binary_model_path = MODELS_DIR / f"{wine_type}_wine_binary_model.pkl"
    scaler_path = MODELS_DIR / f"{wine_type}_wine_scaler.pkl"
    
    # Try to load tuned models first, fall back to regular models if not available
    try:
        regression_model = joblib.load(MODELS_DIR / f"{wine_type}_wine_regression_tuned_model.pkl")
        print("Loaded tuned regression model")
    except FileNotFoundError:
        regression_model = joblib.load(regression_model_path)
        print("Loaded standard regression model")
    
    try:
        binary_model = joblib.load(MODELS_DIR / f"{wine_type}_wine_binary_tuned_model.pkl")
        print("Loaded tuned binary classification model")
    except FileNotFoundError:
        binary_model = joblib.load(binary_model_path)
        print("Loaded standard binary classification model")
    
    # Load scaler
    scaler = joblib.load(scaler_path)
    
    return data, regression_model, binary_model, scaler

def create_partial_dependence_plots(data, model, features, wine_type, model_type="regression"):
    """
    Create and save partial dependence plots for key features.
    
    Args:
        data: DataFrame with wine data
        model: Trained model
        features: List of features to plot
        wine_type: Type of wine ('red', 'white', or 'combined')
        model_type: Type of model ('regression' or 'classification')
    """
    print(f"\nCreating partial dependence plots for {wine_type} wine ({model_type} model)...")
    
    # Prepare data
    X = data.drop(['quality', 'quality_class', 'wine_type', 'wine_type_numeric'], axis=1, errors='ignore')
    
    # Create output directory
    pdp_dir = INTERPRETATION_DIR / f"{wine_type}_pdp"
    os.makedirs(pdp_dir, exist_ok=True)
    
    # Create combined figure for report
    fig, axes = plt.subplots(len(features), 1, figsize=(10, 4 * len(features)))
    
    # Generate PDP for each feature
    for i, feature in enumerate(features):
        # Skip if feature doesn't exist in dataset
        if feature not in X.columns:
            print(f"Feature {feature} not found in data, skipping...")
            continue
            
        # Get feature information
        feature_info = WINE_CHEMISTRY_INSIGHTS.get(feature, {
            'name': feature,
            'description': f"No description available for {feature}",
            'expected_effect': 'unknown',
            'typical_range': {'red': 'unknown', 'white': 'unknown'},
            'critical_threshold': {'red': None, 'white': None},
            'explanation': 'No detailed explanation available'
        })
        
        # Create individual PDP plot
        plt.figure(figsize=(10, 6))
        
        # Calculate partial dependence
        pd_result = partial_dependence(model, X, [feature], grid_resolution=50, kind='average')
        
        # Get x and y values
        feature_values = pd_result["values"][0]
        feature_effect = pd_result["average"][0]
        
        # Add threshold line if available
        critical_value = feature_info.get('critical_threshold', {}).get(wine_type)
        
        # Plot the partial dependence
        plt.plot(feature_values, feature_effect, linewidth=3, color='#1f77b4')
        
        # Add critical threshold if available
        if critical_value is not None:
            plt.axvline(x=critical_value, color='red', linestyle='--', alpha=0.7,
                       label=f'Critical threshold: {critical_value}')
            
            # Annotate regions
            max_y = max(feature_effect)
            min_y = min(feature_effect)
            y_middle = (max_y + min_y) / 2
            
            if feature_info['expected_effect'] == 'positive':
                plt.text(critical_value * 0.9, y_middle, "Lower quality", 
                        ha='right', va='center', color='darkred', fontsize=12, alpha=0.7)
                plt.text(critical_value * 1.1, y_middle, "Higher quality", 
                        ha='left', va='center', color='darkgreen', fontsize=12, alpha=0.7)
            elif feature_info['expected_effect'] == 'negative':
                plt.text(critical_value * 0.9, y_middle, "Higher quality", 
                        ha='right', va='center', color='darkgreen', fontsize=12, alpha=0.7)
                plt.text(critical_value * 1.1, y_middle, "Lower quality", 
                        ha='left', va='center', color='darkred', fontsize=12, alpha=0.7)
            elif feature_info['expected_effect'] == 'optimal range':
                # For optimal range, would need more complex logic with multiple thresholds
                plt.text(critical_value, y_middle, "Optimal threshold", 
                        ha='center', va='bottom', color='darkblue', fontsize=12, alpha=0.7)
        
        # Add typical range
        typical_range = feature_info.get('typical_range', {}).get(wine_type, '').split('-')
        if len(typical_range) == 2:
            try:
                range_min = float(typical_range[0].strip().rstrip('% g/L cm³'))
                range_max = float(typical_range[1].strip().rstrip('% g/L cm³'))
                plt.axvspan(range_min, range_max, alpha=0.2, color='green', label=f'Typical range: {feature_info["typical_range"][wine_type]}')
            except (ValueError, TypeError):
                pass
                
        # Add plot details
        plt.title(f"Effect of {feature_info['name']} on Wine Quality ({wine_type.capitalize()} Wine)\n", fontsize=16)
        plt.xlabel(feature_info['name'], fontsize=14)
        
        if model_type == "regression":
            plt.ylabel("Predicted Quality Score", fontsize=14)
        else:
            plt.ylabel("Probability of 'Good' Quality", fontsize=14)
        
        plt.grid(True, alpha=0.3, linestyle='--')
        plt.legend()
        
        # Add annotation with feature description
        plt.figtext(0.5, 0.01, f"{feature_info['description']}\n\n{feature_info['explanation']}", 
                   ha='center', fontsize=12, bbox={"facecolor":"whitesmoke", "alpha":0.5, "pad":5})
        
        plt.tight_layout()
        plt.savefig(pdp_dir / f"{feature}_{model_type}_pdp.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Add to combined figure
        ax = axes[i] if len(features) > 1 else axes
        ax.plot(feature_values, feature_effect, linewidth=3, color='#1f77b4')
        
        if critical_value is not None:
            ax.axvline(x=critical_value, color='red', linestyle='--', alpha=0.7,
                      label=f'Critical threshold: {critical_value}')
                
        ax.set_title(f"Effect of {feature_info['name']} on Wine Quality", fontsize=12)
        ax.set_xlabel(feature_info['name'], fontsize=10)
        
        if model_type == "regression":
            ax.set_ylabel("Predicted Quality", fontsize=10)
        else:
            ax.set_ylabel("Probability of 'Good'", fontsize=10)
        
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(fontsize=8)
    
    # Save combined figure
    plt.tight_layout()
    plt.savefig(INTERPRETATION_DIR / f"{wine_type}_{model_type}_pdp_combined.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Partial dependence plots saved to {pdp_dir}")

def perform_threshold_analysis(data, regression_model, binary_model, wine_type):
    """
    Perform threshold analysis to identify critical values for key features.
    
    Args:
        data: DataFrame with wine data
        regression_model: Trained regression model
        binary_model: Trained binary classification model
        wine_type: Type of wine ('red', 'white', or 'combined')
    """
    print(f"\nPerforming threshold analysis for {wine_type} wine...")
    
    # Prepare data
    X = data.drop(['quality', 'quality_class', 'wine_type', 'wine_type_numeric'], axis=1, errors='ignore')
    features = KEY_FEATURES[wine_type]
    
    # Create output directory
    threshold_dir = INTERPRETATION_DIR / f"{wine_type}_thresholds"
    os.makedirs(threshold_dir, exist_ok=True)
    
    # Create threshold analysis for each feature
    threshold_results = []
    
    for feature in features:
        # Skip if feature doesn't exist in dataset
        if feature not in X.columns:
            print(f"Feature {feature} not found in data, skipping...")
            continue
            
        print(f"Analyzing thresholds for {feature}...")
        
        # Get feature information
        feature_info = WINE_CHEMISTRY_INSIGHTS.get(feature, {
            'name': feature,
            'expected_effect': 'unknown',
            'critical_threshold': {'red': None, 'white': None}
        })
        
        # Create feature value ranges for analysis
        feature_min = X[feature].min()
        feature_max = X[feature].max()
        step_size = (feature_max - feature_min) / 50
        feature_values = np.arange(feature_min, feature_max, step_size)
        
        # Prepare a dataset where only this feature varies
        X_test = X.iloc[0:1].copy()
        X_test = pd.concat([X_test] * len(feature_values), ignore_index=True)
        
        # Set feature to range of values
        X_test[feature] = feature_values
        
        # Get predictions
        regression_predictions = regression_model.predict(X_test)
        binary_predictions = binary_model.predict_proba(X_test)[:, 1]  # Probability of "good"
        
        # Create figure
        plt.figure(figsize=(12, 8))
        
        # Plot regression predictions
        ax1 = plt.subplot(2, 1, 1)
        ax1.plot(feature_values, regression_predictions, linewidth=3, color='#1f77b4')
        ax1.set_ylabel("Predicted Quality Score", fontsize=12)
        ax1.grid(True, alpha=0.3, linestyle='--')
        
        # Add critical threshold if available
        critical_value = feature_info.get('critical_threshold', {}).get(wine_type)
        if critical_value is not None:
            ax1.axvline(x=critical_value, color='red', linestyle='--', alpha=0.7,
                       label=f'Critical threshold: {critical_value}')
        
        # Calculate rate of change to identify "threshold points"
        rate_of_change = np.gradient(regression_predictions, feature_values)
        
        # Find points where rate of change is significant
        mean_change = np.mean(np.abs(rate_of_change))
        threshold_points = []
        
        for i in range(1, len(rate_of_change) - 1):
            # Check if rate of change is significantly higher than average
            if abs(rate_of_change[i]) > 2 * mean_change:
                # Check if it's a local maximum in rate of change
                if abs(rate_of_change[i]) > abs(rate_of_change[i-1]) and abs(rate_of_change[i]) > abs(rate_of_change[i+1]):
                    threshold_points.append(i)
        
        # Mark threshold points
        for idx in threshold_points:
            ax1.scatter(feature_values[idx], regression_predictions[idx], color='red', s=100, 
                       label=f'Threshold: {feature_values[idx]:.2f}', zorder=5)
        
        # Remove duplicate labels
        handles, labels = ax1.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax1.legend(by_label.values(), by_label.keys(), fontsize=10)
        
        # Plot binary predictions
        ax2 = plt.subplot(2, 1, 2)
        ax2.plot(feature_values, binary_predictions, linewidth=3, color='#2ca02c')
        ax2.set_ylabel("Probability of 'Good' Quality", fontsize=12)
        ax2.set_xlabel(feature_info['name'], fontsize=12)
        ax2.grid(True, alpha=0.3, linestyle='--')
        
        # Add critical threshold if available
        if critical_value is not None:
            ax2.axvline(x=critical_value, color='red', linestyle='--', alpha=0.7,
                       label=f'Critical threshold: {critical_value}')
            
        # Find where probability crosses 0.5
        binary_threshold_points = []
        for i in range(1, len(binary_predictions)):
            if (binary_predictions[i-1] < 0.5 and binary_predictions[i] >= 0.5) or \
               (binary_predictions[i-1] >= 0.5 and binary_predictions[i] < 0.5):
                binary_threshold_points.append(i)
        
        # Mark binary threshold points
        for idx in binary_threshold_points:
            ax2.scatter(feature_values[idx], binary_predictions[idx], color='red', s=100, 
                       label=f'Threshold: {feature_values[idx]:.2f}', zorder=5)
        
        # Remove duplicate labels
        handles, labels = ax2.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax2.legend(by_label.values(), by_label.keys(), fontsize=10)
        
        # Add title
        plt.suptitle(f"Threshold Analysis for {feature_info['name']} ({wine_type.capitalize()} Wine)", fontsize=16)
        
        # Add annotation with feature description
        feature_description = WINE_CHEMISTRY_INSIGHTS.get(feature, {}).get('description', f"No description available for {feature}")
        plt.figtext(0.5, 0.01, feature_description, ha='center', fontsize=12, 
                   bbox={"facecolor":"whitesmoke", "alpha":0.5, "pad":5})
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.9, bottom=0.15)
        plt.savefig(threshold_dir / f"{feature}_threshold_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Collect threshold results
        detected_thresholds = [feature_values[idx] for idx in threshold_points]
        binary_thresholds = [feature_values[idx] for idx in binary_threshold_points]
        
        # Compare with domain knowledge
        domain_threshold = feature_info.get('critical_threshold', {}).get(wine_type)
        
        threshold_results.append({
            'feature': feature,
            'feature_name': feature_info.get('name', feature),
            'detected_thresholds': detected_thresholds,
            'binary_thresholds': binary_thresholds,
            'domain_threshold': domain_threshold,
            'expected_effect': feature_info.get('expected_effect', 'unknown'),
            'typical_range': feature_info.get('typical_range', {}).get(wine_type, 'unknown')
        })
    
    # Create summary report
    with open(INTERPRETATION_DIR / f"{wine_type}_threshold_analysis.md", 'w') as f:
        f.write(f"# Threshold Analysis for {wine_type.capitalize()} Wine Quality\n\n")
        f.write("This report identifies critical threshold values for key features that significantly impact wine quality.\n\n")
        
        f.write("## Summary of Findings\n\n")
        f.write("| Feature | Detected Thresholds | Binary Classification Thresholds | Domain Knowledge Threshold | Expected Effect | Typical Range |\n")
        f.write("| ------- | ------------------- | ------------------------------- | -------------------------- | --------------- | ------------- |\n")
        
        for result in threshold_results:
            detected = ", ".join([f"{v:.2f}" for v in result['detected_thresholds']]) if result['detected_thresholds'] else "None"
            binary = ", ".join([f"{v:.2f}" for v in result['binary_thresholds']]) if result['binary_thresholds'] else "None"
            domain = f"{result['domain_threshold']:.2f}" if result['domain_threshold'] is not None else "Unknown"
            
            f.write(f"| {result['feature_name']} | {detected} | {binary} | {domain} | {result['expected_effect']} | {result['typical_range']} |\n")
        
        f.write("\n## Interpretation\n\n")
        f.write("The threshold values identified above represent points where the feature's value leads to significant changes in predicted wine quality.\n\n")
        
        f.write("### Key Insights\n\n")
        
        for result in threshold_results:
            f.write(f"#### {result['feature_name']}\n\n")
            
            feature_info = WINE_CHEMISTRY_INSIGHTS.get(result['feature'], {})
            
            f.write(f"{feature_info.get('description', 'No description available')}\n\n")
            
            # Compare model-detected thresholds with domain knowledge
            if result['domain_threshold'] is not None:
                model_thresholds = result['detected_thresholds'] + result['binary_thresholds']
                if model_thresholds:
                    closest_threshold = min(model_thresholds, key=lambda x: abs(x - result['domain_threshold']))
                    percent_diff = abs(closest_threshold - result['domain_threshold']) / result['domain_threshold'] * 100
                    
                    if percent_diff < 15:
                        f.write(f"✅ The model-detected threshold ({closest_threshold:.2f}) closely matches domain knowledge ({result['domain_threshold']:.2f}).\n\n")
                    else:
                        f.write(f"⚠️ The model-detected threshold ({closest_threshold:.2f}) differs from domain knowledge ({result['domain_threshold']:.2f}).\n\n")
                else:
                    f.write(f"⚠️ No significant thresholds detected by the model, but domain knowledge suggests {result['domain_threshold']:.2f} as a critical value.\n\n")
            else:
                if result['detected_thresholds'] or result['binary_thresholds']:
                    f.write(f"ℹ️ Model detected thresholds, but no domain knowledge reference is available for comparison.\n\n")
                else:
                    f.write(f"ℹ️ No significant thresholds detected for this feature.\n\n")
            
            # Add explanation
            f.write(f"{feature_info.get('explanation', 'No detailed explanation available.')}\n\n")
            
            # Add recommendation based on expected effect
            if result['expected_effect'] == 'positive':
                f.write("**Recommendation**: Higher values of this feature generally improve wine quality.\n\n")
            elif result['expected_effect'] == 'negative':
                f.write("**Recommendation**: Lower values of this feature generally improve wine quality.\n\n")
            elif result['expected_effect'] == 'optimal range':
                f.write("**Recommendation**: This feature has an optimal range - neither too high nor too low for best quality.\n\n")
            else:
                f.write("**Recommendation**: Impact of this feature depends on other factors and wine style.\n\n")
        
        f.write("\n## Methodology\n\n")
        f.write("Thresholds were identified using two approaches:\n\n")
        f.write("1. **Rate of change analysis**: Points where the predicted quality score changes rapidly as the feature value changes.\n")
        f.write("2. **Binary classification threshold**: Points where the probability of 'good' quality crosses 0.5.\n\n")
        f.write("These were compared with thresholds from wine chemistry domain knowledge to validate our findings.\n")
    
    print(f"Threshold analysis saved to {INTERPRETATION_DIR / f'{wine_type}_threshold_analysis.md'}")

def create_shap_analysis(data, model, wine_type, model_type="regression"):
    """
    Create SHAP analysis to understand feature impact on predictions.
    
    Args:
        data: DataFrame with wine data
        model: Trained model
        wine_type: Type of wine ('red', 'white', or 'combined')
        model_type: Type of model ('regression' or 'classification')
    """
    print(f"\nCreating SHAP analysis for {wine_type} wine ({model_type} model)...")
    
    # Prepare data
    X = data.drop(['quality', 'quality_class', 'wine_type', 'wine_type_numeric'], axis=1, errors='ignore')
    
    # Sample data if too large (for performance)
    if len(X) > 500:
        X = X.sample(500, random_state=42)
    
    # Create output directory
    shap_dir = INTERPRETATION_DIR / f"{wine_type}_shap"
    os.makedirs(shap_dir, exist_ok=True)
    
    try:
        # Create SHAP explainer
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)
        
        # For classification, we need to handle multiple outputs
        if model_type == "classification" and isinstance(shap_values, list):
            # For binary classification, we're interested in class 1 (good wine)
            shap_values = shap_values[1]
        
        # Create summary plot
        plt.figure(figsize=(12, 10))
        shap.summary_plot(shap_values, X, plot_type="bar", show=False)
        plt.title(f"SHAP Feature Importance - {wine_type.capitalize()} Wine ({model_type.capitalize()})", fontsize=16)
        plt.tight_layout()
        plt.savefig(shap_dir / f"{wine_type}_{model_type}_shap_importance.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create summary plot (beeswarm)
        plt.figure(figsize=(14, 12))
        shap.summary_plot(shap_values, X, show=False)
        plt.title(f"SHAP Feature Impact - {wine_type.capitalize()} Wine ({model_type.capitalize()})", fontsize=16)
        plt.tight_layout()
        plt.savefig(shap_dir / f"{wine_type}_{model_type}_shap_summary.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Generate detailed SHAP plots for top features
        feature_importance = np.abs(shap_values).mean(0)
        feature_importance_df = pd.DataFrame({
            'feature': X.columns,
            'importance': feature_importance
        }).sort_values('importance', ascending=False)
        
        top_features = feature_importance_df.head(5)['feature'].tolist()
        
        for feature in top_features:
            plt.figure(figsize=(10, 6))
            shap.dependence_plot(feature, shap_values, X, show=False)
            plt.title(f"SHAP Dependence Plot - {feature} ({wine_type.capitalize()} Wine)", fontsize=16)
            plt.tight_layout()
            plt.savefig(shap_dir / f"{wine_type}_{model_type}_{feature}_shap_dependence.png", dpi=300, bbox_inches='tight')
            plt.close()
        
        print(f"SHAP analysis saved to {shap_dir}")
        
    except Exception as e:
        print(f"Error creating SHAP analysis: {e}")
        print("Skipping SHAP analysis due to error.")

def create_domain_insights_report(wine_types):
    """
    Create a comprehensive domain insights report about wine chemistry and quality.
    
    Args:
        wine_types: List of wine types to include in the report ('red', 'white', 'combined')
    """
    print("\nCreating domain insights report...")
    
    with open(INTERPRETATION_DIR / "wine_chemistry_insights.md", 'w') as f:
        f.write("# Wine Chemistry and Quality: Domain Insights\n\n")
        f.write("This report provides domain-specific insights about the relationship between wine chemistry and quality.\n\n")
        
        f.write("## Chemical Properties and Their Impact on Wine Quality\n\n")
        
        # Loop through all features in the domain knowledge dictionary
        for feature, info in WINE_CHEMISTRY_INSIGHTS.items():
            f.write(f"### {info['name']}\n\n")
            f.write(f"{info['description']}\n\n")
            
            # Add typical ranges
            f.write("**Typical ranges:**\n\n")
            for wine_type in wine_types:
                if wine_type in info.get('typical_range', {}):
                    f.write(f"- {wine_type.capitalize()} wine: {info['typical_range'][wine_type]}\n")
            f.write("\n")
            
            # Add expected effect
            f.write("**Expected effect on quality:** ")
            if info['expected_effect'] == 'positive':
                f.write("Higher values generally correlate with higher quality.\n\n")
            elif info['expected_effect'] == 'negative':
                f.write("Lower values generally correlate with higher quality.\n\n") 
            elif info['expected_effect'] == 'optimal range':
                f.write("Has an optimal range - neither too high nor too low for best quality.\n\n")
            else:
                f.write(f"{info['expected_effect']}.\n\n")
            
            # Add detailed explanation
            f.write("**Detailed explanation:**\n\n")
            f.write(f"{info['explanation']}\n\n")
            
            # Add thresholds
            f.write("**Critical thresholds:**\n\n")
            for wine_type in wine_types:
                if wine_type in info.get('critical_threshold', {}):
                    f.write(f"- {wine_type.capitalize()} wine: {info['critical_threshold'][wine_type]}\n")
            f.write("\n")
            
            # Add practical recommendations
            f.write("**Practical recommendations for winemakers:**\n\n")
            if info['expected_effect'] == 'positive':
                f.write(f"- Target higher levels of {info['name'].lower()} within the typical range\n")
                f.write(f"- Aim for at least {info.get('critical_threshold', {}).get('red', 'N/A')} for red wines\n")
                f.write(f"- Aim for at least {info.get('critical_threshold', {}).get('white', 'N/A')} for white wines\n\n")
            elif info['expected_effect'] == 'negative':
                f.write(f"- Minimize {info['name'].lower()} levels in the wine\n")
                f.write(f"- Keep below {info.get('critical_threshold', {}).get('red', 'N/A')} for red wines\n")
                f.write(f"- Keep below {info.get('critical_threshold', {}).get('white', 'N/A')} for white wines\n\n")
            elif info['expected_effect'] == 'optimal range':
                f.write(f"- Maintain {info['name'].lower()} within the optimal range\n")
                f.write(f"- For red wines, values around {info.get('critical_threshold', {}).get('red', 'N/A')} are ideal\n")
                f.write(f"- For white wines, values around {info.get('critical_threshold', {}).get('white', 'N/A')} are ideal\n\n")
            else:
                f.write(f"- Consider {info['name'].lower()} in the context of wine style\n")
                f.write("- Balance with other chemical components for overall quality\n\n")
        
        # Add general recommendations
        f.write("## General Wine Quality Recommendations\n\n")
        
        f.write("### Red Wine\n\n")
        f.write("For high-quality red wines, focus on:\n\n")
        f.write("1. **Higher alcohol content** (>12%) from riper grapes\n")
        f.write("2. **Lower volatile acidity** (<0.7 g/L) through careful fermentation\n")
        f.write("3. **Moderate sulphates** (0.6-0.8 g/L) for preservation\n")
        f.write("4. **Balanced pH** (3.3-3.6) for stability and flavor\n")
        f.write("5. **Free-to-total SO2 ratio** >0.4 for effective preservation\n\n")
        
        f.write("### White Wine\n\n")
        f.write("For high-quality white wines, focus on:\n\n")
        f.write("1. **Higher alcohol content** (>11.5%) for body and structure\n")
        f.write("2. **Very low volatile acidity** (<0.4 g/L) for freshness\n")
        f.write("3. **Appropriate residual sugar** for style (dry: <4 g/L, off-dry: 4-12 g/L)\n")
        f.write("4. **Lower pH** (3.0-3.3) for crispness and stability\n")
        f.write("5. **Sufficient free SO2** (>35 mg/L) to prevent oxidation\n\n")
        
        f.write("## Interpreting the Model in a Practical Context\n\n")
        f.write("Our machine learning models have learned these relationships from data, often confirming established wine chemistry knowledge. ")
        f.write("When the model and domain knowledge agree, we can be more confident in the recommendations. ")
        f.write("When they disagree, it suggests either unique patterns in the data or limitations in the model.\n\n")
        
        f.write("Winemakers can use these insights to:\n\n")
        f.write("1. **Adjust fermentation conditions** to influence key parameters\n")
        f.write("2. **Guide blending decisions** to optimize chemical profiles\n")
        f.write("3. **Make informed preservative additions** based on threshold analysis\n")
        f.write("4. **Customize approaches for different wine styles** using type-specific thresholds\n")
    
    print(f"Domain insights report saved to {INTERPRETATION_DIR / 'wine_chemistry_insights.md'}")

def main():
    """Main function to execute the model interpretation pipeline."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Perform advanced model interpretation for wine quality prediction.')
    parser.add_argument('--wine-type', type=str, choices=['red', 'white', 'combined'], default='all',
                       help='Type of wine to analyze (default: all)')
    
    args = parser.parse_args()
    
    # Determine wine types to analyze
    if args.wine_type == 'all':
        wine_types = ['red', 'white', 'combined']
    else:
        wine_types = [args.wine_type]
    
    print(f"Starting model interpretation for {', '.join(wine_types)} wine...")
    
    # Loop through each wine type
    for wine_type in wine_types:
        try:
            # Load data and models
            data, regression_model, binary_model, scaler = load_data_and_models(wine_type)
            
            # Create partial dependence plots for regression model
            create_partial_dependence_plots(data, regression_model, KEY_FEATURES[wine_type], wine_type, model_type="regression")
            
            # Create partial dependence plots for binary model
            create_partial_dependence_plots(data, binary_model, KEY_FEATURES[wine_type], wine_type, model_type="binary")
            
            # Perform threshold analysis
            perform_threshold_analysis(data, regression_model, binary_model, wine_type)
            
            # Create SHAP analysis
            create_shap_analysis(data, regression_model, wine_type, model_type="regression")
            create_shap_analysis(data, binary_model, wine_type, model_type="classification")
            
        except Exception as e:
            print(f"Error processing {wine_type} wine: {e}")
            import traceback
            traceback.print_exc()
    
    # Create domain insights report
    create_domain_insights_report(wine_types)
    
    print("\nModel interpretation complete!")
    print(f"Reports and visualizations saved to {INTERPRETATION_DIR}")

if __name__ == "__main__":
    main()