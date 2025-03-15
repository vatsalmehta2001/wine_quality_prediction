"""
Wine Quality Feature Engineering

This script applies feature engineering techniques to the wine quality datasets
to prepare them for machine learning models. It includes:
- Feature scaling/normalization
- Feature transformation
- Feature creation
- Feature selection
- Outlier handling

Usage:
    python src/features/feature_engineering.py

Author: Vatsal Mehta
Date: March 2025
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import joblib

# Define paths
ROOT_DIR = Path(__file__).resolve().parents[2]
PROCESSED_DATA_DIR = ROOT_DIR / "data" / "processed"
ENGINEERED_DATA_DIR = ROOT_DIR / "data" / "processed" / "engineered"
MODELS_DIR = ROOT_DIR / "models"

# Create directories if they don't exist
os.makedirs(ENGINEERED_DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

def load_processed_data():
    """
    Load processed wine quality datasets.
    
    Returns:
        tuple: red_wine_df, white_wine_df, combined_df
    """
    red_wine = pd.read_csv(PROCESSED_DATA_DIR / "winequality_red_processed.csv")
    white_wine = pd.read_csv(PROCESSED_DATA_DIR / "winequality_white_processed.csv")
    combined_wine = pd.read_csv(PROCESSED_DATA_DIR / "winequality_combined_processed.csv")
    
    print(f"Red wine dataset: {red_wine.shape}")
    print(f"White wine dataset: {white_wine.shape}")
    print(f"Combined dataset: {combined_wine.shape}")
    
    return red_wine, white_wine, combined_wine

def handle_outliers(df, columns, method='clip'):
    """
    Handle outliers in the specified columns.
    
    Args:
        df: DataFrame to process
        columns: List of columns to check for outliers
        method: Method to handle outliers ('clip', 'remove', or 'winsorize')
        
    Returns:
        DataFrame with outliers handled
    """
    df_processed = df.copy()
    
    if method == 'clip':
        for col in columns:
            Q1 = df_processed[col].quantile(0.25)
            Q3 = df_processed[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            # Clip values outside the bounds
            df_processed[col] = df_processed[col].clip(lower=lower_bound, upper=upper_bound)
            
    elif method == 'remove':
        for col in columns:
            Q1 = df_processed[col].quantile(0.25)
            Q3 = df_processed[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            # Create mask for values inside the bounds
            mask = (df_processed[col] >= lower_bound) & (df_processed[col] <= upper_bound)
            df_processed = df_processed.loc[mask]
            
    elif method == 'winsorize':
        for col in columns:
            # Replace outliers with the nearest non-outlier value
            Q1 = df_processed[col].quantile(0.05)
            Q3 = df_processed[col].quantile(0.95)
            df_processed[col] = df_processed[col].clip(lower=Q1, upper=Q3)
    
    return df_processed

def create_derived_features(df):
    """
    Create new features based on domain knowledge and existing features.
    
    Args:
        df: DataFrame to process
        
    Returns:
        DataFrame with new features
    """
    df_new = df.copy()
    
    # Total acidity (fixed + volatile)
    df_new['total_acidity'] = df_new['fixed acidity'] + df_new['volatile acidity']
    
    # Acidity to sugar ratio
    df_new['acidity_to_sugar'] = df_new['total_acidity'] / (df_new['residual sugar'] + 1)  # +1 to avoid division by zero
    
    # Free to total sulfur dioxide ratio
    df_new['free_to_total_so2_ratio'] = df_new['free sulfur dioxide'] / (df_new['total sulfur dioxide'] + 1)  # +1 to avoid division by zero
    
    # Alcohol to acidity ratio
    df_new['alcohol_to_acidity'] = df_new['alcohol'] / (df_new['total_acidity'] + 1)  # +1 to avoid division by zero
    
    # Sugar to alcohol ratio (sweetness perception)
    df_new['sugar_to_alcohol'] = df_new['residual sugar'] / (df_new['alcohol'] + 1)  # +1 to avoid division by zero
    
    # Complex feature: balance score (alcohol-acidity-sugar balance)
    df_new['balance_score'] = df_new['alcohol'] / (df_new['total_acidity'] * np.log1p(df_new['residual sugar']))
    
    # Complex feature: body score (alcohol, density)
    df_new['body_score'] = df_new['alcohol'] * (1 - (df_new['density'] - 0.9))
    
    # Convert wine_type to numeric if needed for certain models
    df_new['wine_type_numeric'] = df_new['wine_type'].map({'red': 0, 'white': 1})
    
    return df_new

def apply_feature_transformation(df):
    """
    Apply transformations to skewed features.
    
    Args:
        df: DataFrame to process
        
    Returns:
        DataFrame with transformed features
    """
    df_transformed = df.copy()
    
    # List of features to check for skewness
    numeric_features = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    numeric_features = [col for col in numeric_features if col not in ['quality', 'wine_type_numeric']]
    
    # Calculate skewness and apply transformation to heavily skewed features
    for feature in numeric_features:
        skewness = df_transformed[feature].skew()
        
        # Apply log transformation to positively skewed features
        if skewness > 0.5:
            # Add a small constant to avoid log(0)
            df_transformed[f"{feature}_log"] = np.log1p(df_transformed[feature])
        
        # Apply square root transformation to moderately skewed features
        elif skewness > 0.3:
            df_transformed[f"{feature}_sqrt"] = np.sqrt(df_transformed[feature])
    
    return df_transformed

def scale_features(df, scaler_type='standard'):
    """
    Scale numerical features using specified scaler.
    
    Args:
        df: DataFrame to process
        scaler_type: Type of scaler to use ('standard' or 'minmax')
        
    Returns:
        Tuple of (scaled DataFrame, scaler object)
    """
    # Copy dataframe
    df_scaled = df.copy()
    
    # Select only numeric columns for scaling, excluding target
    numeric_features = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    features_to_scale = [col for col in numeric_features if col not in ['quality', 'wine_type_numeric']]
    
    # Initialize scaler
    if scaler_type == 'standard':
        scaler = StandardScaler()
    else:  # minmax
        scaler = MinMaxScaler()
    
    # Fit and transform
    scaled_features = scaler.fit_transform(df_scaled[features_to_scale])
    
    # Replace original features with scaled versions
    df_scaled[features_to_scale] = scaled_features
    
    return df_scaled, scaler, features_to_scale

def select_features(X, y, method='random_forest', k=10):
    """
    Select the most important features using specified method.
    
    Args:
        X: Feature matrix
        y: Target variable
        method: Feature selection method ('random_forest', 'f_regression', 'mutual_info')
        k: Number of features to select
        
    Returns:
        Tuple of (selected feature names, feature importances)
    """
    # Adjust k if it's larger than the number of features
    k = min(k, X.shape[1])
    
    feature_names = X.columns.tolist()
    
    if method == 'random_forest':
        # Use Random Forest importance
        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        rf.fit(X, y)
        importances = rf.feature_importances_
        
        # Sort features by importance
        indices = np.argsort(importances)[::-1]
        selected_indices = indices[:k]
        
    elif method == 'f_regression':
        # Use F-value between feature and target
        selector = SelectKBest(score_func=f_regression, k=k)
        selector.fit(X, y)
        importances = selector.scores_
        selected_indices = selector.get_support(indices=True)
        
    elif method == 'mutual_info':
        # Use mutual information
        selector = SelectKBest(score_func=mutual_info_regression, k=k)
        selector.fit(X, y)
        importances = selector.scores_
        selected_indices = selector.get_support(indices=True)
        
    else:
        raise ValueError(f"Unknown feature selection method: {method}")
    
    # Get names of selected features
    selected_features = [feature_names[i] for i in selected_indices]
    
    # Get importances for selected features
    selected_importances = {feature_names[i]: importances[i] for i in selected_indices}
    
    return selected_features, selected_importances

def apply_pca(X, n_components=0.95):
    """
    Apply PCA for dimensionality reduction.
    
    Args:
        X: Feature matrix
        n_components: Number of components or variance ratio to preserve
        
    Returns:
        Tuple of (transformed data, PCA object)
    """
    # Standardize data before PCA (important!)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Apply PCA
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)
    
    # Create dataframe with PCA components
    pca_df = pd.DataFrame(
        X_pca, 
        columns=[f'PC{i+1}' for i in range(X_pca.shape[1])]
    )
    
    # Show explained variance
    print(f"Number of PCA components: {pca.n_components_}")
    print(f"Explained variance ratio: {sum(pca.explained_variance_ratio_):.4f}")
    
    return pca_df, pca, scaler

def convert_quality_to_class(df, binary=False, multi_class=False):
    """
    Convert quality score to classification target.
    
    Args:
        df: DataFrame with quality column
        binary: If True, convert to binary classification (good/bad)
        multi_class: If True, convert to multi-class (bad/average/good)
        
    Returns:
        DataFrame with new target column
    """
    df_new = df.copy()
    
    if binary:
        # Binary: bad (0) vs good (1)
        # Quality 6 and above is considered good
        df_new['quality_class'] = (df_new['quality'] >= 6).astype(int)
        print("Converted to binary classification (0=bad, 1=good)")
        
    elif multi_class:
        # Multi-class: bad (0) vs average (1) vs good (2)
        bins = [0, 4, 6, 10]  # Adjust these thresholds as needed
        labels = [0, 1, 2]
        df_new['quality_class'] = pd.cut(df_new['quality'], bins=bins, labels=labels, include_lowest=True)
        df_new['quality_class'] = df_new['quality_class'].astype(int)
        print("Converted to multi-class classification (0=bad, 1=average, 2=good)")
    
    return df_new

def process_and_save_data(red_df, white_df, combined_df):
    """
    Apply feature engineering to datasets and save results.
    
    Args:
        red_df: Red wine dataframe
        white_df: White wine dataframe
        combined_df: Combined wine dataframe
    """
    print("\nProcessing red wine dataset...")
    # Process red wine data
    red_processed = handle_outliers(red_df, red_df.select_dtypes(include=['float64']).columns)
    red_processed = create_derived_features(red_processed)
    red_processed = apply_feature_transformation(red_processed)
    red_scaled, red_scaler, red_scaled_features = scale_features(red_processed, scaler_type='standard')
    
    # Create classification targets
    red_with_classes = convert_quality_to_class(red_scaled, binary=True)
    red_with_classes = convert_quality_to_class(red_with_classes, multi_class=True)
    
    # Split into features and target for red wine
    X_red = red_with_classes.drop(['quality', 'quality_class', 'wine_type'], axis=1)
    y_red = red_with_classes['quality']
    
    # Feature selection for red wine
    red_selected_features, red_importances = select_features(X_red, y_red, method='random_forest', k=15)
    print(f"Selected top {len(red_selected_features)} features for red wine:")
    for feature, importance in sorted(red_importances.items(), key=lambda x: x[1], reverse=True):
        print(f"  {feature}: {importance:.4f}")
    
    # Save red wine processed data
    red_with_classes.to_csv(ENGINEERED_DATA_DIR / "red_wine_engineered.csv", index=False)
    joblib.dump(red_scaler, MODELS_DIR / "red_wine_scaler.pkl")
    
    print("\nProcessing white wine dataset...")
    # Process white wine data
    white_processed = handle_outliers(white_df, white_df.select_dtypes(include=['float64']).columns)
    white_processed = create_derived_features(white_processed)
    white_processed = apply_feature_transformation(white_processed)
    white_scaled, white_scaler, white_scaled_features = scale_features(white_processed, scaler_type='standard')
    
    # Create classification targets
    white_with_classes = convert_quality_to_class(white_scaled, binary=True)
    white_with_classes = convert_quality_to_class(white_with_classes, multi_class=True)
    
    # Split into features and target for white wine
    X_white = white_with_classes.drop(['quality', 'quality_class', 'wine_type'], axis=1)
    y_white = white_with_classes['quality']
    
    # Feature selection for white wine
    white_selected_features, white_importances = select_features(X_white, y_white, method='random_forest', k=15)
    print(f"Selected top {len(white_selected_features)} features for white wine:")
    for feature, importance in sorted(white_importances.items(), key=lambda x: x[1], reverse=True):
        print(f"  {feature}: {importance:.4f}")
    
    # Save white wine processed data
    white_with_classes.to_csv(ENGINEERED_DATA_DIR / "white_wine_engineered.csv", index=False)
    joblib.dump(white_scaler, MODELS_DIR / "white_wine_scaler.pkl")
    
    print("\nProcessing combined dataset...")
    # Process combined data
    combined_processed = handle_outliers(combined_df, combined_df.select_dtypes(include=['float64']).columns)
    combined_processed = create_derived_features(combined_processed)
    combined_processed = apply_feature_transformation(combined_processed)
    combined_scaled, combined_scaler, combined_scaled_features = scale_features(combined_processed, scaler_type='standard')
    
    # Create classification targets
    combined_with_classes = convert_quality_to_class(combined_scaled, binary=True)
    combined_with_classes = convert_quality_to_class(combined_with_classes, multi_class=True)
    
    # Split into features and target for combined data
    X_combined = combined_with_classes.drop(['quality', 'quality_class', 'wine_type'], axis=1)
    y_combined = combined_with_classes['quality']
    
    # Feature selection for combined data
    combined_selected_features, combined_importances = select_features(X_combined, y_combined, method='random_forest', k=15)
    print(f"Selected top {len(combined_selected_features)} features for combined dataset:")
    for feature, importance in sorted(combined_importances.items(), key=lambda x: x[1], reverse=True):
        print(f"  {feature}: {importance:.4f}")
    
    # Apply PCA to combined dataset
    X_pca, pca_model, pca_scaler = apply_pca(X_combined, n_components=0.95)
    
    # Add target back to PCA dataframe
    pca_df = X_pca.copy()
    pca_df['quality'] = y_combined.values
    pca_df['quality_class'] = combined_with_classes['quality_class'].values
    pca_df['wine_type'] = combined_with_classes['wine_type'].values
    
    # Save combined wine processed data
    combined_with_classes.to_csv(ENGINEERED_DATA_DIR / "combined_wine_engineered.csv", index=False)
    pca_df.to_csv(ENGINEERED_DATA_DIR / "combined_wine_pca.csv", index=False)
    joblib.dump(combined_scaler, MODELS_DIR / "combined_wine_scaler.pkl")
    joblib.dump(pca_model, MODELS_DIR / "combined_wine_pca.pkl")
    joblib.dump(pca_scaler, MODELS_DIR / "combined_wine_pca_scaler.pkl")
    
    # Save feature importances
    feature_importance_df = pd.DataFrame({
        'feature': list(combined_importances.keys()),
        'importance': list(combined_importances.values())
    }).sort_values('importance', ascending=False)
    
    feature_importance_df.to_csv(ENGINEERED_DATA_DIR / "feature_importances.csv", index=False)
    
    print("\nFeature engineering complete!")
    print(f"Engineered datasets saved to {ENGINEERED_DATA_DIR}")
    print(f"Models and scalers saved to {MODELS_DIR}")

def main():
    """Main function to execute the feature engineering pipeline."""
    print("Starting feature engineering...")
    
    # Load data
    red_wine_df, white_wine_df, combined_wine_df = load_processed_data()
    
    # Process and save data
    process_and_save_data(red_wine_df, white_wine_df, combined_wine_df)

if __name__ == "__main__":
    main()