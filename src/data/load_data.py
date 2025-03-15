"""
Wine Quality Dataset Loader

This script loads the wine quality datasets (red and white) and performs basic data checks.
It saves the combined dataset to the processed data directory.

Usage:
    python src/data/load_data.py

Author: Vatsal Mehta
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
RAW_DATA_DIR = ROOT_DIR / "data" / "raw"
PROCESSED_DATA_DIR = ROOT_DIR / "data" / "processed"

# Create directories if they don't exist
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

def load_wine_data():
    """
    Load red and white wine datasets, add type indicator, and combine them.
    
    Returns:
        tuple: red_wine_df, white_wine_df, combined_df
    """
    print("Loading wine quality datasets...")
    
    # Load datasets
    try:
        red_wine = pd.read_csv(RAW_DATA_DIR / "winequality-red.csv", sep=';')
        white_wine = pd.read_csv(RAW_DATA_DIR / "winequality-white.csv", sep=';')
        
        # Add wine type column
        red_wine['wine_type'] = 'red'
        white_wine['wine_type'] = 'white'
        
        # Combine datasets for potential combined model
        combined_wine = pd.concat([red_wine, white_wine], axis=0, ignore_index=True)
        
        # Check data loading
        print(f"Red wine dataset shape: {red_wine.shape}")
        print(f"White wine dataset shape: {white_wine.shape}")
        print(f"Combined dataset shape: {combined_wine.shape}")
        
        return red_wine, white_wine, combined_wine
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please make sure wine quality CSV files are in the data/raw directory.")
        return None, None, None

def check_data_quality(red_df, white_df, combined_df):
    """
    Perform basic data quality checks on the datasets.
    
    Args:
        red_df: Red wine dataframe
        white_df: White wine dataframe
        combined_df: Combined wine dataframe
    """
    print("\n--- Data Quality Check ---")
    
    # Check for missing values
    print("\nMissing values in red wine dataset:")
    print(red_df.isnull().sum())
    
    print("\nMissing values in white wine dataset:")
    print(white_df.isnull().sum())
    
    # Check for duplicate entries
    red_duplicates = red_df.duplicated().sum()
    white_duplicates = white_df.duplicated().sum()
    print(f"\nDuplicate entries in red wine dataset: {red_duplicates}")
    print(f"Duplicate entries in white wine dataset: {white_duplicates}")
    
    # Check the quality distribution
    print("\nQuality distribution in red wine dataset:")
    print(red_df['quality'].value_counts().sort_index())
    
    print("\nQuality distribution in white wine dataset:")
    print(white_df['quality'].value_counts().sort_index())
    
    # Check basic statistics
    print("\nRed wine statistics:")
    print(red_df.describe())
    
    print("\nWhite wine statistics:")
    print(white_df.describe())

def save_processed_data(red_df, white_df, combined_df):
    """
    Save preprocessed data to the processed data directory.
    
    Args:
        red_df: Red wine dataframe
        white_df: White wine dataframe
        combined_df: Combined wine dataframe
    """
    print("\nSaving processed datasets...")
    
    red_df.to_csv(PROCESSED_DATA_DIR / "winequality_red_processed.csv", index=False)
    white_df.to_csv(PROCESSED_DATA_DIR / "winequality_white_processed.csv", index=False)
    combined_df.to_csv(PROCESSED_DATA_DIR / "winequality_combined_processed.csv", index=False)
    
    print(f"Datasets saved to {PROCESSED_DATA_DIR}")

def main():
    """Main function to execute the data loading and checking pipeline."""
    # Load data
    red_wine_df, white_wine_df, combined_wine_df = load_wine_data()
    
    if red_wine_df is not None:
        # Check data quality
        check_data_quality(red_wine_df, white_wine_df, combined_wine_df)
        
        # Save processed data
        save_processed_data(red_wine_df, white_wine_df, combined_wine_df)
        
        print("\nData loading and initial processing complete!")

if __name__ == "__main__":
    main()