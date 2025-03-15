"""
Wine Quality Data Visualization

This script creates visualizations for wine quality data to better understand
the relationships between features and target variable, as well as the
distributions of different features.

Usage:
    python src/visualization/explore_visualize.py

Author: Your Name
Date: March 2025
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Define paths
ROOT_DIR = Path(__file__).resolve().parents[2]
PROCESSED_DATA_DIR = ROOT_DIR / "data" / "processed"
REPORTS_DIR = ROOT_DIR / "reports" / "figures"

# Create reports directory if it doesn't exist
os.makedirs(REPORTS_DIR, exist_ok=True)

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

def plot_quality_distribution(red_df, white_df, combined_df):
    """
    Plot the distribution of wine quality scores.
    
    Args:
        red_df: Red wine dataframe
        white_df: White wine dataframe
        combined_df: Combined wine dataframe
    """
    plt.figure(figsize=(12, 5))
    
    # Plot distributions
    plt.subplot(1, 2, 1)
    sns.countplot(x='quality', data=red_df, color='red', alpha=0.7)
    sns.countplot(x='quality', data=white_df, color='skyblue', alpha=0.7)
    plt.title('Wine Quality Distribution by Type')
    plt.xlabel('Quality Score')
    plt.ylabel('Count')
    plt.legend(['Red Wine', 'White Wine'])
    
    # Plot percentage distribution
    plt.subplot(1, 2, 2)
    quality_percentage = combined_df.groupby(['wine_type', 'quality']).size().unstack(0)
    quality_percentage = quality_percentage.div(quality_percentage.sum()).mul(100)
    quality_percentage.plot(kind='bar', stacked=False)
    plt.title('Wine Quality Distribution (%)')
    plt.xlabel('Quality Score')
    plt.ylabel('Percentage')
    
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "quality_distribution.png", dpi=300)
    plt.close()
    
    print("Quality distribution plot saved.")

def plot_feature_distributions(red_df, white_df):
    """
    Create histograms of each feature to compare red and white wines.
    
    Args:
        red_df: Red wine dataframe
        white_df: White wine dataframe
    """
    # Get numerical features
    numerical_features = red_df.select_dtypes(include=[np.number]).columns.tolist()
    numerical_features.remove('quality')  # We'll handle quality separately
    
    # Create multiple histograms
    n_features = len(numerical_features)
    n_cols = 3
    n_rows = (n_features + n_cols - 1) // n_cols
    
    plt.figure(figsize=(15, n_rows * 4))
    
    for i, feature in enumerate(numerical_features):
        plt.subplot(n_rows, n_cols, i + 1)
        
        # Plot normalized histograms for comparison
        sns.histplot(red_df[feature], color='red', alpha=0.5, stat='density', kde=True, label='Red Wine')
        sns.histplot(white_df[feature], color='skyblue', alpha=0.5, stat='density', kde=True, label='White Wine')
        
        plt.title(f'Distribution of {feature}')
        plt.xlabel(feature)
        plt.legend()
    
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "feature_distributions.png", dpi=300)
    plt.close()
    
    print("Feature distributions plot saved.")

def plot_correlation_matrices(red_df, white_df):
    """
    Create correlation heatmaps for red and white wines.
    
    Args:
        red_df: Red wine dataframe
        white_df: White wine dataframe
    """
    # Get numerical features
    numerical_cols = red_df.select_dtypes(include=[np.number]).columns
    
    # Plot correlation matrix for red wine
    plt.figure(figsize=(12, 10))
    plt.subplot(1, 2, 1)
    red_corr = red_df[numerical_cols].corr()
    sns.heatmap(red_corr, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5, vmin=-1, vmax=1)
    plt.title('Red Wine Correlation Matrix')
    
    # Plot correlation matrix for white wine
    plt.subplot(1, 2, 2)
    white_corr = white_df[numerical_cols].corr()
    sns.heatmap(white_corr, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5, vmin=-1, vmax=1)
    plt.title('White Wine Correlation Matrix')
    
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "correlation_matrices.png", dpi=300)
    plt.close()
    
    print("Correlation matrices saved.")

def plot_feature_vs_quality(combined_df):
    """
    Create scatter plots of each feature vs quality, colored by wine type.
    
    Args:
        combined_df: Combined wine dataframe
    """
    # Get numerical features
    numerical_features = combined_df.select_dtypes(include=[np.number]).columns.tolist()
    numerical_features.remove('quality')  # We'll handle quality separately
    
    # Create multiple scatter plots
    n_features = len(numerical_features)
    n_cols = 3
    n_rows = (n_features + n_cols - 1) // n_cols
    
    plt.figure(figsize=(15, n_rows * 4))
    
    for i, feature in enumerate(numerical_features):
        plt.subplot(n_rows, n_cols, i + 1)
        
        # Add jitter to quality for better visualization
        quality_jitter = combined_df['quality'] + np.random.normal(0, 0.1, size=len(combined_df))
        
        # Plot scatter with wine type as color
        sns.scatterplot(
            x=feature, 
            y=quality_jitter, 
            hue='wine_type',
            data=combined_df,
            alpha=0.5
        )
        
        # Add trend line
        sns.regplot(
            x=feature, 
            y='quality', 
            data=combined_df,
            scatter=False,
            color='black'
        )
        
        plt.title(f'{feature} vs Quality')
        plt.xlabel(feature)
        plt.ylabel('Quality Score')
    
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "feature_vs_quality.png", dpi=300)
    plt.close()
    
    print("Feature vs quality plots saved.")

def plot_boxplots_by_quality(combined_df):
    """
    Create boxplots of features grouped by quality score.
    
    Args:
        combined_df: Combined wine dataframe
    """
    # Get numerical features
    numerical_features = combined_df.select_dtypes(include=[np.number]).columns.tolist()
    numerical_features.remove('quality')  # We'll handle quality separately
    
    # Create multiple boxplots
    n_features = len(numerical_features)
    n_cols = 2
    n_rows = (n_features + n_cols - 1) // n_cols
    
    plt.figure(figsize=(15, n_rows * 5))
    
    for i, feature in enumerate(numerical_features):
        plt.subplot(n_rows, n_cols, i + 1)
        
        # Plot boxplot with quality as x-axis
        sns.boxplot(
            x='quality', 
            y=feature, 
            hue='wine_type',
            data=combined_df
        )
        
        plt.title(f'{feature} by Quality')
        plt.xlabel('Quality Score')
        plt.ylabel(feature)
    
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "boxplots_by_quality.png", dpi=300)
    plt.close()
    
    print("Boxplots by quality saved.")

def plot_pairplot(red_df, white_df):
    """
    Create pairplots for selected features of red and white wines.
    
    Args:
        red_df: Red wine dataframe
        white_df: White wine dataframe
    """
    # Ensure quality is numeric
    red_df = red_df.copy()
    white_df = white_df.copy()
    
    # Convert quality to string for hue to prevent issues
    red_df['quality_str'] = red_df['quality'].astype(str)
    white_df['quality_str'] = white_df['quality'].astype(str)
    
    # Select most important features based on correlation with quality
    # Only include numeric columns in correlation calculation
    numeric_cols = red_df.select_dtypes(include=[np.number]).columns
    numeric_cols = [col for col in numeric_cols if col != 'quality']
    
    red_corr = red_df[numeric_cols + ['quality']].corr()['quality'].abs().sort_values(ascending=False)
    white_corr = white_df[numeric_cols + ['quality']].corr()['quality'].abs().sort_values(ascending=False)
    
    # Get top 5 features for each wine type (not including quality/wine_type)
    red_top_features = red_corr.index[:5].tolist()
    white_top_features = white_corr.index[:5].tolist()
    
    # Create combined features list
    top_features = list(set(red_top_features + white_top_features))
    
    try:
        # Create pairplot for red wine
        plt.figure(figsize=(12, 10))
        sns.pairplot(
            red_df[top_features + ['quality_str']], 
            hue='quality_str',
            palette='RdBu',
            diag_kind='kde'
        )
        plt.suptitle('Red Wine - Pairplot of Top Features', y=1.02)
        plt.tight_layout()
        plt.savefig(REPORTS_DIR / "red_wine_pairplot.png", dpi=300)
        plt.close()
        
        # Create pairplot for white wine
        plt.figure(figsize=(12, 10))
        sns.pairplot(
            white_df[top_features + ['quality_str']], 
            hue='quality_str',
            palette='RdBu',
            diag_kind='kde'
        )
        plt.suptitle('White Wine - Pairplot of Top Features', y=1.02)
        plt.tight_layout()
        plt.savefig(REPORTS_DIR / "white_wine_pairplot.png", dpi=300)
        plt.close()
        
        print("Pairplots saved.")
    except Exception as e:
        print(f"Error creating pairplots: {e}")
        # Alternative visualization if pairplot fails
        print("Creating alternative correlation heatmaps instead...")
        
        # Create correlation plots for top features
        plt.figure(figsize=(10, 8))
        sns.heatmap(red_df[top_features + ['quality']].corr(), 
                    annot=True, 
                    cmap='coolwarm',
                    vmin=-1,
                    vmax=1)
        plt.title('Red Wine - Correlation of Top Features')
        plt.tight_layout()
        plt.savefig(REPORTS_DIR / "red_wine_correlation.png", dpi=300)
        plt.close()
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(white_df[top_features + ['quality']].corr(), 
                    annot=True, 
                    cmap='coolwarm',
                    vmin=-1,
                    vmax=1)
        plt.title('White Wine - Correlation of Top Features')
        plt.tight_layout()
        plt.savefig(REPORTS_DIR / "white_wine_correlation.png", dpi=300)
        plt.close()

def main():
    """Main function to execute the visualization pipeline."""
    print("Starting data visualization...")
    
    # Load data
    red_wine_df, white_wine_df, combined_wine_df = load_processed_data()
    
    # Create visualizations
    plot_quality_distribution(red_wine_df, white_wine_df, combined_wine_df)
    plot_feature_distributions(red_wine_df, white_wine_df)
    plot_correlation_matrices(red_wine_df, white_wine_df)
    plot_feature_vs_quality(combined_wine_df)
    plot_boxplots_by_quality(combined_wine_df)
    plot_pairplot(red_wine_df, white_wine_df)
    
    print("Data visualization complete!")
    print(f"All visualizations saved to {REPORTS_DIR}")

if __name__ == "__main__":
    # Set Seaborn style
    sns.set(style="whitegrid")
    sns.set_context("paper")
    
    # Run main function
    main()