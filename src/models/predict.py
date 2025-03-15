"""
Wine Quality Prediction

This script loads trained models and makes predictions on new wine samples.
It can be used both as a command-line tool and imported as a module.

Usage:
    # As a command-line tool
    python src/models/predict.py --wine-type red --task regression --input-file new_samples.csv
    
    # As a module
    from src.models.predict import WineQualityPredictor
    predictor = WineQualityPredictor(wine_type='red', task='binary')
    prediction = predictor.predict(features)

Author: Your Name
Date: March 2025
"""

import os
import argparse
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

# Define paths
ROOT_DIR = Path(__file__).resolve().parents[2]
MODELS_DIR = ROOT_DIR / "models"

def create_derived_features(df):
    """
    Create new features based on domain knowledge and existing features.
    This should match the feature engineering process used during training.
    
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
    
    # Apply log transformations
    for col in df_new.columns:
        if col not in ['wine_type', 'quality', 'quality_class']:
            # Add log transformation for positive features
            df_new[f"{col}_log"] = np.log1p(df_new[col] + 1e-5)  # Adding a small constant to avoid log(0)
            
            # Add square root transformation
            df_new[f"{col}_sqrt"] = np.sqrt(df_new[col] + 1e-5)  # Adding a small constant to avoid sqrt of negative
    
    return df_new

class WineQualityPredictor:
    """Class for making wine quality predictions using trained models."""
    
    def __init__(self, wine_type='red', task='regression', model_path=None):
        """
        Initialize the predictor.
        
        Args:
            wine_type: Type of wine ('red', 'white', or 'combined')
            task: Prediction task ('regression', 'binary', or 'multiclass')
            model_path: Path to custom model file (optional)
        """
        self.wine_type = wine_type
        self.task = task
        
        # Load the model
        if model_path:
            self.model_path = model_path
        else:
            self.model_path = MODELS_DIR / f"{wine_type}_wine_{task}_model.pkl"
        
        try:
            self.model = joblib.load(self.model_path)
            print(f"Model loaded from {self.model_path}")
        except FileNotFoundError:
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        
        # Load the scaler if available
        self.scaler_path = MODELS_DIR / f"{wine_type}_wine_scaler.pkl"
        if os.path.exists(self.scaler_path):
            self.scaler = joblib.load(self.scaler_path)
            print(f"Scaler loaded from {self.scaler_path}")
        else:
            self.scaler = None
            print("No scaler found, using raw features")
    
    def predict(self, features):
        """
        Make prediction for a single sample or batch of samples.
        
        Args:
            features: DataFrame or array of features
            
        Returns:
            Predictions
        """
        # Convert to DataFrame if it's an array or list
        if isinstance(features, (list, np.ndarray)):
            features = pd.DataFrame([features])
        
        # Apply feature engineering
        features_engineered = create_derived_features(features)
        
        # Get feature names from the model if available
        model_features = None
        if hasattr(self.model, 'feature_names_in_'):
            model_features = self.model.feature_names_in_
        
        # Filter features to match model's expected features if needed
        if model_features is not None:
            missing_features = set(model_features) - set(features_engineered.columns)
            extra_features = set(features_engineered.columns) - set(model_features)
            
            if missing_features:
                # Print warning about missing features
                print(f"Warning: Missing {len(missing_features)} features. Adding with zeros.")
                # Add missing features with zeros
                for feature in missing_features:
                    features_engineered[feature] = 0
            
            # Select only the features the model expects
            features_engineered = features_engineered[model_features]
        
        # Apply scaler if available
        if self.scaler:
            features_scaled = self.scaler.transform(features_engineered)
            predictions = self.model.predict(features_scaled)
        else:
            predictions = self.model.predict(features_engineered)
        
        # Format output based on task type
        if self.task == 'regression':
            # Round to nearest 0.5 since wine quality usually follows this pattern
            predictions = np.round(predictions * 2) / 2
        elif self.task == 'binary':
            predictions = ['Good' if p == 1 else 'Bad' for p in predictions]
        elif self.task == 'multiclass':
            quality_mapping = {0: 'Bad', 1: 'Average', 2: 'Good'}
            predictions = [quality_mapping[p] for p in predictions]
        
        return predictions
    
    def predict_proba(self, features):
        """
        Get prediction probabilities for classification models.
        
        Args:
            features: DataFrame or array of features
            
        Returns:
            Prediction probabilities or None if regression
        """
        if self.task == 'regression':
            print("Probability prediction not available for regression models")
            return None
        
        # Convert to DataFrame if it's an array or list
        if isinstance(features, (list, np.ndarray)):
            features = pd.DataFrame([features])
        
        # Apply feature engineering
        features_engineered = create_derived_features(features)
        
        # Get feature names from the model if available
        model_features = None
        if hasattr(self.model, 'feature_names_in_'):
            model_features = self.model.feature_names_in_
        
        # Filter features to match model's expected features if needed
        if model_features is not None:
            missing_features = set(model_features) - set(features_engineered.columns)
            extra_features = set(features_engineered.columns) - set(model_features)
            
            if missing_features:
                # Add missing features with zeros
                for feature in missing_features:
                    features_engineered[feature] = 0
            
            # Select only the features the model expects
            features_engineered = features_engineered[model_features]
        
        # Apply scaler if available
        if self.scaler:
            features_scaled = self.scaler.transform(features_engineered)
            probabilities = self.model.predict_proba(features_scaled)
        else:
            probabilities = self.model.predict_proba(features_engineered)
        
        return probabilities

def predict_from_csv(file_path, wine_type, task, output_file=None):
    """
    Make predictions for samples in a CSV file.
    
    Args:
        file_path: Path to CSV file with samples
        wine_type: Type of wine ('red', 'white', or 'combined')
        task: Prediction task ('regression', 'binary', or 'multiclass')
        output_file: Path to save predictions (optional)
        
    Returns:
        DataFrame with predictions
    """
    # Load the data
    try:
        data = pd.read_csv(file_path)
        print(f"Loaded {len(data)} samples from {file_path}")
    except Exception as e:
        print(f"Error loading file: {e}")
        return None
    
    # Initialize predictor
    try:
        predictor = WineQualityPredictor(wine_type=wine_type, task=task)
    except Exception as e:
        print(f"Error initializing predictor: {e}")
        return None
    
    # Make predictions
    try:
        predictions = predictor.predict(data)
        
        # Add predictions to data
        if task == 'regression':
            data['predicted_quality'] = predictions
        else:
            data['predicted_class'] = predictions
            
            # Add probabilities for classification tasks
            probas = predictor.predict_proba(data)
            if probas is not None:
                if task == 'binary':
                    data['probability_good'] = probas[:, 1]
                else:  # multiclass
                    for i, class_name in enumerate(['bad', 'average', 'good']):
                        data[f'probability_{class_name}'] = probas[:, i]
        
        # Save to output file if specified
        if output_file:
            data.to_csv(output_file, index=False)
            print(f"Predictions saved to {output_file}")
        
        return data
        
    except Exception as e:
        print(f"Error making predictions: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description='Predict wine quality using trained models')
    parser.add_argument('--wine-type', type=str, choices=['red', 'white', 'combined'], 
                        default='red', help='Type of wine')
    parser.add_argument('--task', type=str, choices=['regression', 'binary', 'multiclass'], 
                        default='regression', help='Prediction task')
    parser.add_argument('--input-file', type=str, required=True, 
                        help='Path to CSV file with samples')
    parser.add_argument('--output-file', type=str, 
                        help='Path to save predictions (optional)')
    parser.add_argument('--model-path', type=str, 
                        help='Path to custom model file (optional)')
    
    args = parser.parse_args()
    
    if args.model_path:
        # Initialize predictor with custom model
        predictor = WineQualityPredictor(
            wine_type=args.wine_type, 
            task=args.task,
            model_path=args.model_path
        )
        
        # Load the data
        try:
            data = pd.read_csv(args.input_file)
            print(f"Loaded {len(data)} samples from {args.input_file}")
        except Exception as e:
            print(f"Error loading file: {e}")
            return
        
        # Make predictions
        predictions = predictor.predict(data)
        
        # Add predictions to data
        if args.task == 'regression':
            data['predicted_quality'] = predictions
        else:
            data['predicted_class'] = predictions
        
        # Save to output file if specified
        if args.output_file:
            data.to_csv(args.output_file, index=False)
            print(f"Predictions saved to {args.output_file}")
        else:
            print("Predictions:")
            print(data)
            
    else:
        # Use the CSV prediction function
        result = predict_from_csv(
            args.input_file, 
            args.wine_type, 
            args.task, 
            args.output_file
        )
        
        if result is not None and args.output_file is None:
            print("Predictions:")
            print(result)

if __name__ == "__main__":
    main()