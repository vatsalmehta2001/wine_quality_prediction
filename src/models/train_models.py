"""
Wine Quality Model Training

This script trains and evaluates various machine learning models for wine quality prediction.
It handles both regression (predicting exact quality score) and classification approaches.

Usage:
    python src/models/train_models.py

Author: Vatsal Mehta
Date: March 2025
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import joblib
import time
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, StratifiedKFold, KFold
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report, confusion_matrix
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVR, SVC
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder

# Define paths
ROOT_DIR = Path(__file__).resolve().parents[2]
ENGINEERED_DATA_DIR = ROOT_DIR / "data" / "processed" / "engineered"
MODELS_DIR = ROOT_DIR / "models"
REPORTS_DIR = ROOT_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# Create directories if they don't exist
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

def load_engineered_data():
    """
    Load engineered wine quality datasets.
    
    Returns:
        tuple: red_wine_df, white_wine_df, combined_df
    """
    red_wine = pd.read_csv(ENGINEERED_DATA_DIR / "red_wine_engineered.csv")
    white_wine = pd.read_csv(ENGINEERED_DATA_DIR / "white_wine_engineered.csv")
    combined_wine = pd.read_csv(ENGINEERED_DATA_DIR / "combined_wine_engineered.csv")
    
    print(f"Red wine dataset: {red_wine.shape}")
    print(f"White wine dataset: {white_wine.shape}")
    print(f"Combined dataset: {combined_wine.shape}")
    
    return red_wine, white_wine, combined_wine

def prepare_data(df, target_type='regression', wine_type=None):
    """
    Prepare data for modeling by splitting features and target.
    
    Args:
        df: DataFrame with wine data
        target_type: 'regression', 'binary', or 'multiclass'
        wine_type: Type of wine ('red', 'white', or None for combined)
        
    Returns:
        tuple: X, y, feature_names
    """
    # Remove non-feature columns based on target type
    drop_columns = ['wine_type']
    
    if wine_type is not None:
        # Filter by wine type if specified
        df = df[df['wine_type'] == wine_type]
    
    if target_type == 'regression':
        target_col = 'quality'
        drop_columns.extend(['quality_class'])
    elif target_type == 'binary':
        target_col = 'quality_class'
        drop_columns.extend(['quality'])
    elif target_type == 'multiclass':
        target_col = 'quality_class'
        drop_columns.extend(['quality'])
    else:
        raise ValueError(f"Unknown target type: {target_type}")
    
    # Get target
    y = df[target_col]
    
    # Get features (drop both wine_type and wine_type_numeric to avoid duplication)
    drop_columns.append('wine_type_numeric')
    
    # Drop columns that might not exist
    drop_columns = [col for col in drop_columns if col in df.columns]
    
    # Get features
    X = df.drop(drop_columns + [target_col], axis=1)
    
    # Get feature names
    feature_names = X.columns.tolist()
    
    return X, y, feature_names

def train_regression_models(X_train, X_test, y_train, y_test, feature_names, wine_type='red'):
    """
    Train and evaluate regression models for wine quality prediction.
    
    Args:
        X_train, X_test: Training and test feature sets
        y_train, y_test: Training and test target values
        feature_names: List of feature names
        wine_type: Type of wine ('red', 'white', or 'combined')
        
    Returns:
        dict: Trained models and their performance metrics
    """
    print(f"\nTraining regression models for {wine_type} wine...")
    
    # Define regression models to train
    models = {
        'Linear Regression': LinearRegression(),
        'Ridge Regression': Ridge(alpha=1.0),
        'Lasso Regression': Lasso(alpha=0.1),
        'ElasticNet': ElasticNet(alpha=0.1, l1_ratio=0.5),
        'Decision Tree': DecisionTreeRegressor(random_state=42),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
        'SVR': SVR(kernel='rbf', C=1.0, epsilon=0.1),
        'KNN': KNeighborsRegressor(n_neighbors=5)
    }
    
    # Dictionary to store results
    results = {}
    
    # Train and evaluate each model
    for name, model in models.items():
        try:
            start_time = time.time()
            
            # Train model
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_test, y_pred)
            
            # Cross-validation score
            cv = KFold(n_splits=5, shuffle=True, random_state=42)
            cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='neg_mean_squared_error')
            cv_rmse = np.sqrt(-cv_scores.mean())
            
            # Training time
            train_time = time.time() - start_time
            
            # Store results
            results[name] = {
                'model': model,
                'mse': mse,
                'rmse': rmse,
                'r2': r2,
                'cv_rmse': cv_rmse,
                'train_time': train_time
            }
            
            # Print results
            print(f"{name}:")
            print(f"  RMSE: {rmse:.4f}")
            print(f"  R²: {r2:.4f}")
            print(f"  CV RMSE: {cv_rmse:.4f}")
            print(f"  Training time: {train_time:.2f} seconds")
            
            # Plot feature importance if available
            if hasattr(model, 'feature_importances_'):
                feature_importance_plot(model, feature_names, wine_type, name.replace(' ', '_').lower())
                
        except Exception as e:
            print(f"Error training {name}: {e}")
    
    # Save best model
    if results:
        best_model_name = min(results, key=lambda x: results[x]['rmse'])
        best_model = results[best_model_name]['model']
        
        print(f"\nBest regression model for {wine_type} wine: {best_model_name}")
        print(f"RMSE: {results[best_model_name]['rmse']:.4f}")
        
        # Save model
        joblib.dump(best_model, MODELS_DIR / f"{wine_type}_wine_regression_model.pkl")
        
        # Create a performance comparison DataFrame
        performance_df = pd.DataFrame({
            'Model': list(results.keys()),
            'RMSE': [results[model]['rmse'] for model in results],
            'R²': [results[model]['r2'] for model in results],
            'CV RMSE': [results[model]['cv_rmse'] for model in results],
            'Training Time (s)': [results[model]['train_time'] for model in results]
        })
        
        # Save performance metrics
        performance_df.to_csv(REPORTS_DIR / f"{wine_type}_wine_regression_performance.csv", index=False)
        
        # Plot model comparison
        plt.figure(figsize=(12, 6))
        
        # Plot RMSE comparison
        plt.subplot(1, 2, 1)
        sns.barplot(x='RMSE', y='Model', data=performance_df.sort_values('RMSE'))
        plt.title(f'RMSE Comparison - {wine_type.capitalize()} Wine')
        plt.xlabel('RMSE (lower is better)')
        
        # Plot R² comparison
        plt.subplot(1, 2, 2)
        sns.barplot(x='R²', y='Model', data=performance_df.sort_values('R²', ascending=False))
        plt.title(f'R² Comparison - {wine_type.capitalize()} Wine')
        plt.xlabel('R² (higher is better)')
        
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / f"{wine_type}_wine_regression_comparison.png", dpi=300)
        plt.close()
    
    return results

def train_classification_models(X_train, X_test, y_train, y_test, feature_names, wine_type='red', multiclass=False):
    """
    Train and evaluate classification models for wine quality prediction.
    
    Args:
        X_train, X_test: Training and test feature sets
        y_train, y_test: Training and test target values
        feature_names: List of feature names
        wine_type: Type of wine ('red', 'white', or 'combined')
        multiclass: Whether to treat as multiclass classification
        
    Returns:
        dict: Trained models and their performance metrics
    """
    class_type = "multiclass" if multiclass else "binary"
    print(f"\nTraining {class_type} classification models for {wine_type} wine...")
    
    # Check if there are enough classes for multiclass classification
    if multiclass:
        n_classes = len(np.unique(y_train))
        if n_classes < 3:
            print(f"Warning: Only {n_classes} classes found in y_train. Multiclass may not be appropriate.")
    
    # Define classification models to train
    models = {
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
        'SVC': SVC(probability=True, random_state=42),
        'KNN': KNeighborsClassifier(n_neighbors=5)
    }
    
    # Dictionary to store results
    results = {}
    
    # Train and evaluate each model
    for name, model in models.items():
        try:
            start_time = time.time()
            
            # Train model
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            
            # Cross-validation score
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy')
            cv_accuracy = cv_scores.mean()
            
            # Training time
            train_time = time.time() - start_time
            
            # Store results
            results[name] = {
                'model': model,
                'accuracy': accuracy,
                'cv_accuracy': cv_accuracy,
                'train_time': train_time,
                'classification_report': classification_report(y_test, y_pred, output_dict=True),
                'confusion_matrix': confusion_matrix(y_test, y_pred)
            }
            
            # Print results
            print(f"{name}:")
            print(f"  Accuracy: {accuracy:.4f}")
            print(f"  CV Accuracy: {cv_accuracy:.4f}")
            print(f"  Training time: {train_time:.2f} seconds")
            
            # Plot feature importance if available
            if hasattr(model, 'feature_importances_'):
                feature_importance_plot(model, feature_names, wine_type, 
                                       f"{class_type}_{name.replace(' ', '_').lower()}")
                
        except Exception as e:
            print(f"Error training {name}: {e}")
    
    # Save best model
    if results:
        best_model_name = max(results, key=lambda x: results[x]['accuracy'])
        best_model = results[best_model_name]['model']
        
        print(f"\nBest {class_type} classification model for {wine_type} wine: {best_model_name}")
        print(f"Accuracy: {results[best_model_name]['accuracy']:.4f}")
        
        # Save model
        joblib.dump(best_model, MODELS_DIR / f"{wine_type}_wine_{class_type}_model.pkl")
        
        # Create a performance comparison DataFrame
        performance_df = pd.DataFrame({
            'Model': list(results.keys()),
            'Accuracy': [results[model]['accuracy'] for model in results],
            'CV Accuracy': [results[model]['cv_accuracy'] for model in results],
            'Training Time (s)': [results[model]['train_time'] for model in results]
        })
        
        # Save performance metrics
        performance_df.to_csv(REPORTS_DIR / f"{wine_type}_wine_{class_type}_performance.csv", index=False)
        
        # Plot model comparison
        plt.figure(figsize=(10, 6))
        
        # Plot accuracy comparison
        sns.barplot(x='Accuracy', y='Model', data=performance_df.sort_values('Accuracy', ascending=False))
        plt.title(f'{class_type.capitalize()} Classification Accuracy - {wine_type.capitalize()} Wine')
        plt.xlabel('Accuracy (higher is better)')
        
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / f"{wine_type}_wine_{class_type}_comparison.png", dpi=300)
        plt.close()
        
        # Plot confusion matrix for best model
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            results[best_model_name]['confusion_matrix'], 
            annot=True, 
            fmt='d',
            cmap='Blues',
            cbar=False
        )
        plt.title(f'Confusion Matrix - {best_model_name} ({wine_type.capitalize()} Wine)')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / f"{wine_type}_wine_{class_type}_confusion_matrix.png", dpi=300)
        plt.close()
    
    return results

def tune_best_model(X_train, X_test, y_train, y_test, wine_type, task_type):
    """
    Perform hyperparameter tuning for Random Forest model.
    
    Args:
        X_train, X_test: Training and test feature sets
        y_train, y_test: Training and test target values
        wine_type: Type of wine ('red', 'white', or 'combined')
        task_type: 'regression', 'binary', or 'multiclass'
        
    Returns:
        dict: Tuned model and its performance metrics
    """
    print(f"\nTuning Random Forest model for {wine_type} wine ({task_type})...")
    
    # Define model and parameter grid based on task type
    if task_type == 'regression':
        # Random Forest is often a good performer for regression
        model = RandomForestRegressor(random_state=42)
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2]
        }
        scoring = 'neg_mean_squared_error'
    else:  # classification
        # Random Forest is often a good performer for classification
        model = RandomForestClassifier(random_state=42)
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2]
        }
        scoring = 'accuracy'
    
    # Perform grid search
    start_time = time.time()
    
    # Use appropriate cross-validation strategy
    if task_type == 'regression':
        cv = KFold(n_splits=5, shuffle=True, random_state=42)
    else:
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    grid_search = GridSearchCV(
        model, 
        param_grid, 
        cv=cv, 
        scoring=scoring, 
        n_jobs=-1, 
        verbose=1
    )
    
    try:
        grid_search.fit(X_train, y_train)
        
        # Get best model
        best_model = grid_search.best_estimator_
        
        # Calculate metrics
        if task_type == 'regression':
            y_pred = best_model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_test, y_pred)
            
            print(f"Tuned Random Forest Regressor:")
            print(f"  Best parameters: {grid_search.best_params_}")
            print(f"  RMSE: {rmse:.4f}")
            print(f"  R²: {r2:.4f}")
            
            # Save metrics
            metrics = {
                'model': best_model,
                'best_params': grid_search.best_params_,
                'rmse': rmse,
                'r2': r2,
                'train_time': time.time() - start_time
            }
            
        else:  # classification
            y_pred = best_model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            
            print(f"Tuned Random Forest Classifier:")
            print(f"  Best parameters: {grid_search.best_params_}")
            print(f"  Accuracy: {accuracy:.4f}")
            
            # Save metrics
            metrics = {
                'model': best_model,
                'best_params': grid_search.best_params_,
                'accuracy': accuracy,
                'classification_report': classification_report(y_test, y_pred, output_dict=True),
                'confusion_matrix': confusion_matrix(y_test, y_pred),
                'train_time': time.time() - start_time
            }
        
        # Save tuned model
        joblib.dump(best_model, MODELS_DIR / f"{wine_type}_wine_{task_type}_tuned_model.pkl")
        
        # Save results
        with open(REPORTS_DIR / f"{wine_type}_wine_{task_type}_tuning_results.txt", 'w') as f:
            f.write(f"Best parameters: {grid_search.best_params_}\n")
            f.write(f"Grid search CV results:\n")
            for param, score in zip(grid_search.cv_results_['params'], grid_search.cv_results_['mean_test_score']):
                f.write(f"{param}: {score}\n")
        
        return metrics
    
    except Exception as e:
        print(f"Error during hyperparameter tuning: {e}")
        return None

def feature_importance_plot(model, features, wine_type, model_name):
    """
    Create feature importance plot for a trained model.
    
    Args:
        model: Trained model with feature_importances_ attribute
        features: List of feature names
        wine_type: Type of wine ('red', 'white', or 'combined')
        model_name: Name of the model for the filename
    """
    # Check if model has feature_importances_ attribute
    if not hasattr(model, 'feature_importances_'):
        return
    
    # Get feature importances
    importances = model.feature_importances_
    
    # Sort features by importance
    indices = np.argsort(importances)[::-1]
    
    # Take top 15 features or all if less than 15
    n_features = min(15, len(features))
    top_indices = indices[:n_features]
    top_features = [features[i] for i in top_indices]
    top_importances = importances[top_indices]
    
    # Plot feature importances
    plt.figure(figsize=(10, 6))
    sns.barplot(x=top_importances, y=top_features)
    plt.title(f'Feature Importance - {wine_type.capitalize()} Wine ({model_name})')
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / f"{wine_type}_wine_{model_name}_feature_importance.png", dpi=300)
    plt.close()

def main():
    """Main function to execute the model training pipeline."""
    print("Starting model training...")
    
    # Load data
    red_wine_df, white_wine_df, combined_wine_df = load_engineered_data()
    
    # Dictionary to store results
    results = {
        'red': {},
        'white': {},
        'combined': {}
    }
    
    # Train models for red wine
    print("\n--- Red Wine Models ---")
    
    # Regression
    X, y, features = prepare_data(red_wine_df, target_type='regression')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    reg_results = train_regression_models(X_train, X_test, y_train, y_test, features, wine_type='red')
    results['red']['regression'] = reg_results
    
    # Tune regression model
    tuned_reg_results = tune_best_model(X_train, X_test, y_train, y_test, 'red', 'regression')
    if tuned_reg_results:
        results['red']['tuned_regression'] = tuned_reg_results
    
    # Binary classification
    X, y, features = prepare_data(red_wine_df, target_type='binary')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    bin_results = train_classification_models(X_train, X_test, y_train, y_test, features, wine_type='red')
    results['red']['binary'] = bin_results
    
    # Tune binary classification model
    tuned_bin_results = tune_best_model(X_train, X_test, y_train, y_test, 'red', 'binary')
    if tuned_bin_results:
        results['red']['tuned_binary'] = tuned_bin_results
    
    # Multiclass classification
    X, y, features = prepare_data(red_wine_df, target_type='multiclass')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    multi_results = train_classification_models(X_train, X_test, y_train, y_test, features, wine_type='red', multiclass=True)
    results['red']['multiclass'] = multi_results
    
    # Tune multiclass classification model
    tuned_multi_results = tune_best_model(X_train, X_test, y_train, y_test, 'red', 'multiclass')
    if tuned_multi_results:
        results['red']['tuned_multiclass'] = tuned_multi_results
    
    # Train models for white wine
    print("\n--- White Wine Models ---")
    
    # Regression
    X, y, features = prepare_data(white_wine_df, target_type='regression')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    reg_results = train_regression_models(X_train, X_test, y_train, y_test, features, wine_type='white')
    results['white']['regression'] = reg_results
    
    # Tune regression model
    tuned_reg_results = tune_best_model(X_train, X_test, y_train, y_test, 'white', 'regression')
    if tuned_reg_results:
        results['white']['tuned_regression'] = tuned_reg_results
    
    # Binary classification
    X, y, features = prepare_data(white_wine_df, target_type='binary')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    bin_results = train_classification_models(X_train, X_test, y_train, y_test, features, wine_type='white')
    results['white']['binary'] = bin_results
    
    # Tune binary classification model
    tuned_bin_results = tune_best_model(X_train, X_test, y_train, y_test, 'white', 'binary')
    if tuned_bin_results:
        results['white']['tuned_binary'] = tuned_bin_results
    
    # Multiclass classification
    X, y, features = prepare_data(white_wine_df, target_type='multiclass')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    multi_results = train_classification_models(X_train, X_test, y_train, y_test, features, wine_type='white', multiclass=True)
    results['white']['multiclass'] = multi_results
    
    # Tune multiclass classification model
    tuned_multi_results = tune_best_model(X_train, X_test, y_train, y_test, 'white', 'multiclass')
    if tuned_multi_results:
        results['white']['tuned_multiclass'] = tuned_multi_results
    
    # Train models for combined dataset
    print("\n--- Combined Wine Models ---")
    
    # Regression
    X, y, features = prepare_data(combined_wine_df, target_type='regression')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    reg_results = train_regression_models(X_train, X_test, y_train, y_test, features, wine_type='combined')
    results['combined']['regression'] = reg_results
    
    # Binary classification
    X, y, features = prepare_data(combined_wine_df, target_type='binary')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    bin_results = train_classification_models(X_train, X_test, y_train, y_test, features, wine_type='combined')
    results['combined']['binary'] = bin_results
    
    # Multiclass classification
    X, y, features = prepare_data(combined_wine_df, target_type='multiclass')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    multi_results = train_classification_models(X_train, X_test, y_train, y_test, features, wine_type='combined', multiclass=True)
    results['combined']['multiclass'] = multi_results
    
    print("\nModel training complete!")
    print(f"Models saved to {MODELS_DIR}")
    print(f"Performance metrics saved to {REPORTS_DIR}")
    print(f"Visualizations saved to {FIGURES_DIR}")
    
    # Create README.md for models directory
    create_models_readme()

def create_models_readme():
    """Create a README.md file for the models directory."""
    with open(MODELS_DIR / "README.md", 'w') as f:
        f.write("# Wine Quality Prediction Models\n\n")
        f.write("This directory contains trained machine learning models for wine quality prediction.\n\n")
        
        f.write("## Model Files\n\n")
        f.write("### Red Wine Models\n\n")
        f.write("- `red_wine_regression_model.pkl`: Predicts the exact quality score (1-10) for red wines\n")
        f.write("- `red_wine_binary_model.pkl`: Classifies red wines as good (1) or bad (0)\n")
        f.write("- `red_wine_multiclass_model.pkl`: Classifies red wines as bad (0), average (1), or good (2)\n")
        f.write("- `red_wine_regression_tuned_model.pkl`: Tuned model for regression prediction\n")
        f.write("- `red_wine_binary_tuned_model.pkl`: Tuned model for binary classification\n")
        f.write("- `red_wine_multiclass_tuned_model.pkl`: Tuned model for multiclass classification\n\n")
        
        f.write("### White Wine Models\n\n")
        f.write("- `white_wine_regression_model.pkl`: Predicts the exact quality score (1-10) for white wines\n")
        f.write("- `white_wine_binary_model.pkl`: Classifies white wines as good (1) or bad (0)\n")
        f.write("- `white_wine_multiclass_model.pkl`: Classifies white wines as bad (0), average (1), or good (2)\n")
        f.write("- `white_wine_regression_tuned_model.pkl`: Tuned model for regression prediction\n")
        f.write("- `white_wine_binary_tuned_model.pkl`: Tuned model for binary classification\n")
        f.write("- `white_wine_multiclass_tuned_model.pkl`: Tuned model for multiclass classification\n\n")
        
        f.write("### Combined Wine Models\n\n")
        f.write("- `combined_wine_regression_model.pkl`: Predicts quality score for any wine (red or white)\n")
        f.write("- `combined_wine_binary_model.pkl`: Classifies any wine as good (1) or bad (0)\n")
        f.write("- `combined_wine_multiclass_model.pkl`: Classifies any wine as bad (0), average (1), or good (2)\n\n")
        
        f.write("## Usage\n\n")
        f.write("To use these models for prediction:\n\n")
        f.write("```python\n")
        f.write("import joblib\n\n")
        f.write("# Load the model\n")
        f.write("model = joblib.load('models/red_wine_binary_model.pkl')\n\n")
        f.write("# Prepare input features (ensure they match the training features)\n")
        f.write("features = [...]\n\n")
        f.write("# Make prediction\n")
        f.write("prediction = model.predict([features])\n")
        f.write("```\n\n")
        
        f.write("## Model Performance\n\n")
        f.write("See the `reports/` directory for detailed model performance metrics and comparisons.\n")

if __name__ == "__main__":
    main()