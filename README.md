# Wine Quality Prediction

A machine learning project to predict wine quality based on physicochemical properties.

## Project Overview

The objective of this project is to make a prediction about the quality of wine (red and white) based on different physicochemical properties. This one showcases the complete end-to-end machine learning workflow from data exploration to model deployment, with the Portuguese Vinho Verde wine dataset.

### Key Features

- **All Data Analysis**: Analysis of wine chemicals and their properties and relation to quality
- **Feature Engineering**: Development of new features that lead to greater accuracy
- **Multiple Modeling Approaches**:
  - Models Regressing Against Exact Quality Score (1-10)
  - Binary classification functions to classify wines as good or bad
  - Multi-class classification models to classify wines as bad, average, or good
- **Model Optimization**: Utilizing HyperParameter tuning to try and improve prediction accuracy
- **Model Comparison**: Testing different algorithms to see which one comes out on top
- **Separate Models**: One per red, one per white, one combined prediction

## Directory Structure

```
wine_quality_prediction/
│
├── data/
│   ├── raw/                    # Original, immutable data
│   └── processed/              # Cleaned and processed data
│       └── engineered/         # Feature-engineered data
│
├── models/                     # Trained and serialized models
│
├── reports/                    # Generated analysis reports
│   └── figures/                # Generated graphics and figures
│
├── src/                        # Source code
│   ├── data/                   # Scripts to download or generate data
│   ├── features/               # Scripts for feature engineering
│   ├── models/                 # Scripts to train and use models
│   └── visualization/          # Scripts for data visualization
│
├── requirements.txt            # Dependencies
└── README.md                   # Project documentation
```

## Installation

1. Clone this repository
```bash
git clone https://github.com/vatsalmehta2001/wine_quality_prediction
cd wine_quality_prediction
```

2. Install required packages
```bash
pip install -r requirements.txt
```

## Usage

### Data Processing

1. Load and explore the data
```bash
python src/data/load_data.py
```

2. Visualize the data
```bash
python src/visualization/explore_visualize.py
```

3. Generate engineered features
```bash
python src/features/feature_engineering.py
```

### Model Training

Train and evaluate various models
```bash
python src/models/train_models.py
```

### Making Predictions

Use trained models to make predictions on new samples
```bash
python src/models/predict.py --wine-type red --task regression --input-file new_samples.csv --output-file predictions.csv
```

## Results

The project achieved the following results:

- **Red Wine**: 
  - Best regression model: Random Forest (RMSE: ~0.56)
  - Best classification accuracy: Gradient Boosting (~90%)

- **White Wine**: 
  - Best regression model: Random Forest (RMSE: ~0.51)
  - Best classification accuracy: Random Forest (~91%)

- **Key Insights**:
  - Alcohol content is one of the most important indicators of wine quality
  - Higher quality wines tend to have higher alcohol content and lower volatile acidity
  - Different chemical properties impact red and white wine quality differently

## Models

The project includes the following trained models:

- Regression models to predict exact quality score
- Binary classification models to predict whether a wine is good or bad
- Multi-class classification models to categorize wines into three quality levels

## Technologies Used

- **Python**: Primary programming language
- **Pandas & NumPy**: Data processing
- **Scikit-learn**: Machine learning algorithms
- **Matplotlib & Seaborn**: Data visualization
- **Joblib**: Model serialization

## Dataset

The datasets used in this project contain physicochemical properties and quality ratings for Portuguese "Vinho Verde" wines. The dataset includes:

- Red wine: 1,599 samples
- White wine: 4,898 samples
- 11 input features (chemical properties)
- 1 output feature (quality score between 0-10)

Reference:
P. Cortez, A. Cerdeira, F. Almeida, T. Matos and J. Reis.
Modeling wine preferences by data mining from physicochemical properties.
In Decision Support Systems, Elsevier, 47(4):547-553, 2009.

## Future Work

- Develop a web application for wine quality prediction
- Add more advanced feature engineering techniques
- Explore deep learning approaches
- Include additional wine types beyond Vinho Verde

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Author

Vatsal Gagankumar Mehta