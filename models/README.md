# Wine Quality Prediction Models

This directory contains trained machine learning models for wine quality prediction.

## Model Files

### Red Wine Models

- `red_wine_regression_model.pkl`: Predicts the exact quality score (1-10) for red wines
- `red_wine_binary_model.pkl`: Classifies red wines as good (1) or bad (0)
- `red_wine_multiclass_model.pkl`: Classifies red wines as bad (0), average (1), or good (2)
- `red_wine_regression_tuned_model.pkl`: Tuned model for regression prediction
- `red_wine_binary_tuned_model.pkl`: Tuned model for binary classification
- `red_wine_multiclass_tuned_model.pkl`: Tuned model for multiclass classification

### White Wine Models

- `white_wine_regression_model.pkl`: Predicts the exact quality score (1-10) for white wines
- `white_wine_binary_model.pkl`: Classifies white wines as good (1) or bad (0)
- `white_wine_multiclass_model.pkl`: Classifies white wines as bad (0), average (1), or good (2)
- `white_wine_regression_tuned_model.pkl`: Tuned model for regression prediction
- `white_wine_binary_tuned_model.pkl`: Tuned model for binary classification
- `white_wine_multiclass_tuned_model.pkl`: Tuned model for multiclass classification

### Combined Wine Models

- `combined_wine_regression_model.pkl`: Predicts quality score for any wine (red or white)
- `combined_wine_binary_model.pkl`: Classifies any wine as good (1) or bad (0)
- `combined_wine_multiclass_model.pkl`: Classifies any wine as bad (0), average (1), or good (2)

## Usage

To use these models for prediction:

```python
import joblib

# Load the model
model = joblib.load('models/red_wine_binary_model.pkl')

# Prepare input features (ensure they match the training features)
features = [...]

# Make prediction
prediction = model.predict([features])
```

## Model Performance

See the `reports/` directory for detailed model performance metrics and comparisons.
