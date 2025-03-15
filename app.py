"""
Wine Quality Prediction Web Interface

This Streamlit app provides a user-friendly interface for predicting wine quality
based on chemical properties. It uses the pre-trained models from the wine_quality_prediction
project.

Usage:
    streamlit run app.py

Author: Your Name
Date: March 2025
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os
import sys

# Add the src directory to the path so we can import our modules
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from src.models.predict import WineQualityPredictor
from src.features.feature_engineering import create_derived_features

# Define paths - Fixed to use the current directory
ROOT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = ROOT_DIR / "models"
REPORTS_DIR = ROOT_DIR / "reports"

# Set page config
st.set_page_config(
    page_title="Wine Quality Predictor",
    page_icon="🍷",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Define wine type options
WINE_TYPES = ["red", "white", "combined"]

# Define prediction task options
PREDICTION_TASKS = {
    "Exact Quality Score (1-10)": "regression",
    "Good/Bad Classification": "binary",
    "Quality Category (Bad/Average/Good)": "multiclass"
}

# Define features and their ranges based on wine chemistry domain knowledge
FEATURE_RANGES = {
    "fixed acidity": {"red": (4.6, 15.9), "white": (3.8, 14.2), "default": (3.8, 15.9), "step": 0.1, "format": "%.1f"},
    "volatile acidity": {"red": (0.12, 1.58), "white": (0.08, 1.1), "default": (0.08, 1.58), "step": 0.01, "format": "%.2f"},
    "citric acid": {"red": (0.0, 1.0), "white": (0.0, 1.66), "default": (0.0, 1.66), "step": 0.01, "format": "%.2f"},
    "residual sugar": {"red": (0.9, 15.5), "white": (0.6, 65.8), "default": (0.6, 65.8), "step": 0.1, "format": "%.1f"},
    "chlorides": {"red": (0.012, 0.611), "white": (0.009, 0.346), "default": (0.009, 0.611), "step": 0.001, "format": "%.3f"},
    "free sulfur dioxide": {"red": (1, 72), "white": (2, 289), "default": (1, 289), "step": 1, "format": "%d"},
    "total sulfur dioxide": {"red": (6, 289), "white": (9, 440), "default": (6, 440), "step": 1, "format": "%d"},
    "density": {"red": (0.990, 1.004), "white": (0.987, 1.039), "default": (0.987, 1.039), "step": 0.0001, "format": "%.4f"},
    "pH": {"red": (2.74, 4.01), "white": (2.72, 3.82), "default": (2.72, 4.01), "step": 0.01, "format": "%.2f"},
    "sulphates": {"red": (0.33, 2.0), "white": (0.22, 1.08), "default": (0.22, 2.0), "step": 0.01, "format": "%.2f"},
    "alcohol": {"red": (8.4, 14.9), "white": (8.0, 14.2), "default": (8.0, 14.9), "step": 0.1, "format": "%.1f"}
}

# Define default values based on average quality wine profiles
DEFAULT_VALUES = {
    "red": {
        "fixed acidity": 8.3,
        "volatile acidity": 0.53,
        "citric acid": 0.27,
        "residual sugar": 2.5,
        "chlorides": 0.087,
        "free sulfur dioxide": 16,
        "total sulfur dioxide": 46,
        "density": 0.997,
        "pH": 3.31,
        "sulphates": 0.66,
        "alcohol": 10.4
    },
    "white": {
        "fixed acidity": 6.9,
        "volatile acidity": 0.28,
        "citric acid": 0.33,
        "residual sugar": 6.4,
        "chlorides": 0.046,
        "free sulfur dioxide": 35,
        "total sulfur dioxide": 138,
        "density": 0.994,
        "pH": 3.19,
        "sulphates": 0.49,
        "alcohol": 10.5
    },
    "combined": {
        "fixed acidity": 7.2,
        "volatile acidity": 0.34,
        "citric acid": 0.32,
        "residual sugar": 5.4,
        "chlorides": 0.056,
        "free sulfur dioxide": 30,
        "total sulfur dioxide": 115,
        "density": 0.995,
        "pH": 3.22,
        "sulphates": 0.53,
        "alcohol": 10.5
    }
}

# Feature descriptions for tooltips
FEATURE_DESCRIPTIONS = {
    "fixed acidity": "Non-volatile acids that contribute to wine structure and taste (g/L, as tartaric acid)",
    "volatile acidity": "Volatile acids that can lead to an unpleasant vinegar taste (g/L, as acetic acid)",
    "citric acid": "Contributes to freshness and flavor in wines (g/L)",
    "residual sugar": "Amount of sugar remaining after fermentation (g/L)",
    "chlorides": "Salt content in the wine (g/L)",
    "free sulfur dioxide": "Free form of SO₂ that prevents microbial growth and oxidation (mg/L)",
    "total sulfur dioxide": "Total amount of SO₂ in the wine (mg/L)",
    "density": "Density of the wine (g/cm³)",
    "pH": "Describes how acidic or basic the wine is (0-14)",
    "sulphates": "Additives that contribute to SO₂ levels and act as antimicrobials/antioxidants (g/L)",
    "alcohol": "Alcohol content (% by volume)"
}

def load_images(wine_type, feature, task_type="regression"):
    """
    Attempt to load PDP and threshold analysis images for selected feature.
    
    Args:
        wine_type: Type of wine ('red', 'white', or 'combined')
        feature: Selected feature name
        task_type: Type of prediction task ('regression', 'binary', 'multiclass')
        
    Returns:
        tuple: (pdp_image_path, threshold_image_path)
    """
    # Try to find PDP image
    pdp_image_path = None
    pdp_dir = REPORTS_DIR / "interpretation" / f"{wine_type}_pdp"
    
    if pdp_dir.exists():
        # First try exact feature name
        potential_pdp_path = pdp_dir / f"{feature}_{task_type}_pdp.png"
        if potential_pdp_path.exists():
            pdp_image_path = potential_pdp_path
    
    # Try to find threshold analysis image
    threshold_image_path = None
    threshold_dir = REPORTS_DIR / "interpretation" / f"{wine_type}_thresholds"
    
    if threshold_dir.exists():
        potential_threshold_path = threshold_dir / f"{feature}_threshold_analysis.png"
        if potential_threshold_path.exists():
            threshold_image_path = potential_threshold_path
    
    return pdp_image_path, threshold_image_path

def create_feature_input_ui(wine_type):
    """
    Create the feature input UI based on the selected wine type.
    
    Args:
        wine_type: Selected wine type ('red', 'white', or 'combined')
        
    Returns:
        dict: Dictionary of feature values entered by the user
    """
    st.markdown("### Wine Chemical Properties")
    st.markdown("Adjust the sliders to set the chemical properties of your wine sample.")
    
    feature_values = {}
    
    # Create two columns for feature inputs
    col1, col2 = st.columns(2)
    
    # Add sliders for each feature
    features = list(FEATURE_RANGES.keys())
    half_point = len(features) // 2
    
    with col1:
        for feature in features[:half_point]:
            min_val, max_val = FEATURE_RANGES[feature].get(wine_type, FEATURE_RANGES[feature]["default"])
            step = FEATURE_RANGES[feature]["step"]
            format_str = FEATURE_RANGES[feature]["format"]
            default_val = DEFAULT_VALUES[wine_type][feature]
            
            feature_values[feature] = st.slider(
                f"{feature}",
                min_value=float(min_val),
                max_value=float(max_val),
                value=float(default_val),
                step=float(step),
                format=format_str,
                help=FEATURE_DESCRIPTIONS.get(feature, "")
            )
    
    with col2:
        for feature in features[half_point:]:
            min_val, max_val = FEATURE_RANGES[feature].get(wine_type, FEATURE_RANGES[feature]["default"])
            step = FEATURE_RANGES[feature]["step"]
            format_str = FEATURE_RANGES[feature]["format"]
            default_val = DEFAULT_VALUES[wine_type][feature]
            
            feature_values[feature] = st.slider(
                f"{feature}",
                min_value=float(min_val),
                max_value=float(max_val),
                value=float(default_val),
                step=float(step),
                format=format_str,
                help=FEATURE_DESCRIPTIONS.get(feature, "")
            )
    
    return feature_values

def make_prediction(wine_type, task_type, feature_values):
    """
    Make a prediction using the selected model.
    
    Args:
        wine_type: Type of wine ('red', 'white', or 'combined')
        task_type: Type of prediction task ('regression', 'binary', 'multiclass')
        feature_values: Dictionary of feature values
        
    Returns:
        tuple: (prediction result, prediction probabilities if applicable)
    """
    try:
        # Create a dataframe with the feature values
        data = pd.DataFrame([feature_values])
        
        # Initialize predictor with custom path to fix potential issues
        model_path = MODELS_DIR / f"{wine_type}_wine_{task_type}_model.pkl"
        if not model_path.exists():
            # Try tuned model as fallback
            model_path = MODELS_DIR / f"{wine_type}_wine_{task_type}_tuned_model.pkl"
            
        if not model_path.exists():
            st.error(f"Model file not found: {model_path}")
            return None, None
            
        predictor = WineQualityPredictor(
            wine_type=wine_type, 
            task=task_type,
            model_path=str(model_path)
        )
        
        # Make prediction
        prediction = predictor.predict(data)[0]
        
        # Get probabilities for classification tasks
        probabilities = None
        if task_type in ['binary', 'multiclass']:
            probabilities = predictor.predict_proba(data)[0]
        
        return prediction, probabilities
    
    except Exception as e:
        st.error(f"Error making prediction: {e}")
        st.exception(e)  # This will show the full traceback in the app
        return None, None

def display_prediction_results(prediction, probabilities, task_type):
    """
    Display the prediction results in a visually appealing format.
    
    Args:
        prediction: Prediction result
        probabilities: Prediction probabilities (if applicable)
        task_type: Type of prediction task ('regression', 'binary', 'multiclass')
    """
    if prediction is None:
        return
    
    st.markdown("### Prediction Results")
    
    if task_type == 'regression':
        # Display quality score with a gauge
        st.markdown(f"<h1 style='text-align: center; color: #2c3e50;'>Predicted Quality: {prediction}</h1>", unsafe_allow_html=True)
        
        # Create gauge chart for quality score
        fig, ax = plt.subplots(figsize=(10, 2))
        
        # Create a horizontal gauge from 0 to 10
        ax.barh(0, 10, color='lightgray', height=0.5)
        ax.barh(0, prediction, color='#3498db', height=0.5)
        
        # Add score markings
        for i in range(11):
            ax.text(i, -0.25, str(i), ha='center', va='center', fontsize=10)
        
        # Add quality labels
        ax.text(2, 0.7, 'Poor', ha='center', va='center', fontsize=10, color='#e74c3c')
        ax.text(5, 0.7, 'Average', ha='center', va='center', fontsize=10, color='#f39c12')
        ax.text(8, 0.7, 'Excellent', ha='center', va='center', fontsize=10, color='#2ecc71')
        
        # Set limits and remove axes
        ax.set_xlim(0, 10)
        ax.set_ylim(-0.5, 1)
        ax.axis('off')
        
        st.pyplot(fig)
        
    elif task_type == 'binary':
        # Display good/bad classification result
        if prediction == "Good":
            emoji = "🍷"
            color = "#2ecc71"
        else:
            emoji = "⚠️"
            color = "#e74c3c"
        
        st.markdown(f"<h1 style='text-align: center; color: {color};'>{emoji} Wine Quality: {prediction}</h1>", 
                    unsafe_allow_html=True)
        
        # Display probability bars
        if probabilities is not None:
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("<h3 style='text-align: center;'>Probability Breakdown</h3>", unsafe_allow_html=True)
                
                # Create bar chart for probabilities
                fig, ax = plt.subplots(figsize=(6, 4))
                labels = ['Bad', 'Good']
                ax.bar(labels, probabilities, color=['#e74c3c', '#2ecc71'])
                ax.set_ylim(0, 1)
                ax.set_ylabel('Probability')
                ax.set_title('Classification Probabilities')
                
                # Add value labels
                for i, v in enumerate(probabilities):
                    ax.text(i, v + 0.02, f'{v:.2f}', ha='center', fontsize=12)
                
                st.pyplot(fig)
                
            with col2:
                st.markdown("<h3 style='text-align: center;'>Confidence Level</h3>", unsafe_allow_html=True)
                
                # Display confidence as a gauge
                confidence = max(probabilities)
                
                # Create confidence meter
                fig, ax = plt.subplots(figsize=(6, 4))
                
                # Create a horizontal gauge from 0 to 1
                ax.barh(0, 1, color='lightgray', height=0.5)
                ax.barh(0, confidence, color='#3498db', height=0.5)
                
                # Add score markings
                for i in np.arange(0, 1.1, 0.2):
                    ax.text(i, -0.25, f'{i:.1f}', ha='center', va='center', fontsize=10)
                
                # Add confidence labels
                ax.text(0.2, 0.7, 'Low', ha='center', va='center', fontsize=10, color='#e74c3c')
                ax.text(0.5, 0.7, 'Medium', ha='center', va='center', fontsize=10, color='#f39c12')
                ax.text(0.8, 0.7, 'High', ha='center', va='center', fontsize=10, color='#2ecc71')
                
                # Set limits and remove axes
                ax.set_xlim(0, 1)
                ax.set_ylim(-0.5, 1)
                ax.set_title(f'Prediction Confidence: {confidence:.2f}')
                ax.set_facecolor('white')
                
                st.pyplot(fig)
    
    elif task_type == 'multiclass':
        # Display multi-class classification result
        if prediction == "Good":
            emoji = "🍷"
            color = "#2ecc71"
        elif prediction == "Average":
            emoji = "🥂"
            color = "#f39c12"
        else:
            emoji = "⚠️"
            color = "#e74c3c"
        
        st.markdown(f"<h1 style='text-align: center; color: {color};'>{emoji} Wine Quality: {prediction}</h1>", 
                   unsafe_allow_html=True)
        
        # Display probability bars if available
        if probabilities is not None:
            st.markdown("<h3 style='text-align: center;'>Probability Breakdown</h3>", unsafe_allow_html=True)
            
            # Create bar chart for probabilities
            fig, ax = plt.subplots(figsize=(8, 4))
            labels = ['Bad', 'Average', 'Good']
            colors = ['#e74c3c', '#f39c12', '#2ecc71']
            ax.bar(labels, probabilities, color=colors)
            ax.set_ylim(0, 1)
            ax.set_ylabel('Probability')
            ax.set_title('Classification Probabilities')
            
            # Add value labels
            for i, v in enumerate(probabilities):
                ax.text(i, v + 0.02, f'{v:.2f}', ha='center', fontsize=12)
            
            st.pyplot(fig)

def display_feature_visualizations(wine_type, feature, task_type):
    """
    Display visualizations related to the selected feature.
    
    Args:
        wine_type: Type of wine ('red', 'white', or 'combined')
        feature: Selected feature name
        task_type: Type of prediction task ('regression', 'binary', 'multiclass')
    """
    pdp_image_path, threshold_image_path = load_images(wine_type, feature, task_type)
    
    if pdp_image_path or threshold_image_path:
        st.markdown("### Feature Visualizations")
        st.markdown("These visualizations show how the selected feature impacts wine quality predictions.")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if pdp_image_path and pdp_image_path.exists():
                st.image(str(pdp_image_path), caption=f"Partial Dependence Plot: {feature}", use_column_width=True)
            else:
                st.info(f"No partial dependence plot available for {feature}")
        
        with col2:
            if threshold_image_path and threshold_image_path.exists():
                st.image(str(threshold_image_path), caption=f"Threshold Analysis: {feature}", use_column_width=True)
            else:
                st.info(f"No threshold analysis available for {feature}")

def main():
    # Add header and description
    st.title("🍷 Wine Quality Predictor")
    st.markdown("""
    This application predicts the quality of wine based on its physicochemical properties.
    Adjust the parameters below to see how different properties impact the predicted quality.
    """)
    
    # Create sidebar for model selection
    st.sidebar.header("Model Selection")
    
    wine_type = st.sidebar.selectbox(
        "Wine Type",
        options=WINE_TYPES,
        format_func=lambda x: x.capitalize(),
        help="Select the type of wine you want to predict"
    )
    
    task_type_name = st.sidebar.selectbox(
        "Prediction Task",
        options=list(PREDICTION_TASKS.keys()),
        help="Choose what type of prediction you want to make"
    )
    task_type = PREDICTION_TASKS[task_type_name]
    
    # Add feature selection for visualization
    st.sidebar.header("Feature Visualization")
    feature_to_visualize = st.sidebar.selectbox(
        "Select Feature to Visualize",
        options=list(FEATURE_RANGES.keys()),
        help="View how this feature impacts wine quality"
    )
    
    # Add information section
    st.sidebar.header("About")
    st.sidebar.info("""
    This application uses machine learning models trained on the 
    Wine Quality dataset from UCI Machine Learning Repository.
    
    The models predict wine quality based on physicochemical 
    properties measured by laboratory tests.
    """)
    
    # Get feature values from UI
    feature_values = create_feature_input_ui(wine_type)
    
    # Display feature visualizations
    display_feature_visualizations(wine_type, feature_to_visualize, task_type)
    
    # Add a button to make prediction
    if st.button("Predict Quality", type="primary"):
        with st.spinner("Making prediction..."):
            # Make prediction
            prediction, probabilities = make_prediction(wine_type, task_type, feature_values)
            
            # Display prediction results
            display_prediction_results(prediction, probabilities, task_type)
    
    # Expander for debugging info
    with st.expander("Debug Information"):
        st.write("Current working directory:", os.getcwd())
        st.write("ROOT_DIR:", ROOT_DIR)
        st.write("MODELS_DIR exists:", MODELS_DIR.exists())
        st.write("REPORTS_DIR exists:", REPORTS_DIR.exists())
        
        # List available model files
        if MODELS_DIR.exists():
            st.write("Available model files:")
            model_files = list(MODELS_DIR.glob("*.pkl"))
            for model_file in model_files:
                st.write(f"- {model_file.name}")
    
    # Add footer
    st.markdown("---")
    st.markdown("© 2025 Wine Quality Prediction Project | Made with Streamlit")

if __name__ == "__main__":
    main()