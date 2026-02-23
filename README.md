# Orange Telecom Churn Prediction

A comprehensive machine learning project to predict customer churn for Orange Telecom using advanced classification algorithms with probability calibration and optimized decision thresholds.

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Dataset](#dataset)
- [Technologies](#technologies)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Methodology](#methodology)
- [Results](#results)
- [Future Improvements](#future-improvements)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)
- [Acknowledgments](#acknowledgments)

## Overview

Customer churn is a critical metric for telecom companies and many businesses today. This project analyzes customer data to predict which customers are likely to leave, understand user behavior, and provide actionable insights for retention strategies.

This analysis uses the Orange Telecom dataset from [Kaggle](https://www.kaggle.com/datasets/mnassrib/telecom-churn-datasets), a cleaned version of the [KDD Cup 2009](https://www.kdd.org/kdd-cup/view/kdd-cup-2009) dataset, which uses "databases from the French Telecom company Orange to predict the propensity of customers to switch provider".

## Key Features

- **Gold Standard ML Workflow**: Strict train/test isolation with test set touched only once
- **Advanced Preprocessing**: Custom transformers preventing cross-validation data leakage
- **Probability Calibration**: CalibratedClassifierCV with Platt Scaling for reliable probability estimates
- **Threshold Optimization**: Precision-recall analysis targeting 80% recall for business requirements
- **Comprehensive Model Comparison**: DummyClassifier, Logistic Regression, Random Forest, Gradient Boosting, and XGBoost
- **Business Impact Analysis**: ROI calculations and actionable customer retention lists

## Dataset

- **Source**: [Orange Telecom Customer Churn Dataset (Kaggle)](https://www.kaggle.com/datasets/mnassrib/telecom-churn-datasets)
- **Original Source**: [KDD Cup 2009](https://www.kdd.org/kdd-cup/view/kdd-cup-2009) - French Telecom company Orange
- **Size**: 3,333 customer records (combined from provided train/test files)
- **Features**: 20 variables including customer demographics, usage patterns, and service information
- **Target**: Binary classification (Churn: True/False)
- **Class Distribution**: Approximately 85% No Churn / 15% Churn (imbalanced)
- **Split Strategy**: Custom 80/20 stratified train/test split (random_state=42)

## Technologies

- **Python 3.8+**
- **Data Analysis**: pandas, numpy
- **Visualization**: matplotlib, seaborn
- **Machine Learning**: scikit-learn, XGBoost
- **Development Environment**: Jupyter Notebook

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager
- Kaggle account (for dataset download)

### Setup Instructions

1. Clone the repository:
```bash
git clone https://github.com/jaden-lai/orange-telecom-churn-prediction.git
cd orange-telecom-churn-prediction
```

2. Download the dataset from Kaggle:
   - Visit the [Orange Telecom Churn Dataset on Kaggle](https://www.kaggle.com/datasets/mnassrib/telecom-churn-datasets)
   - Click the "Download" button to download the dataset (requires Kaggle login)
   - Extract the downloaded ZIP file
   - Place the following CSV files in the `data/` folder:
     - `churn-bigml-80.csv`
     - `churn-bigml-20.csv`
   
   Your directory structure should look like:
   ```
   orange-telecom-churn-prediction/
   ├── data/
   │   ├── churn-bigml-80.csv
   │   ├── churn-bigml-20.csv
   │   └── README.md
   └── ...
   ```

3. Install required packages:
```bash
pip install -r requirements.txt
```

4. Verify installation:
```bash
python -c "import pandas, numpy, sklearn, xgboost; print('All dependencies installed successfully')"
```

## Usage

1. Launch Jupyter Notebook:
```bash
jupyter notebook
```

2. Open `orange_churn_analysis.ipynb` to view the complete analysis

3. Run cells sequentially from top to bottom to reproduce the analysis

4. The notebook is organized into clear phases:
   - Phase 1: Environment Setup & Data Loading
   - Phase 2: Exploratory Data Analysis (EDA)
   - Phase 3: Feature Engineering & Preprocessing
   - Phase 4: Model Training & Cross-Validation Comparison
   - Phase 5: Model Calibration & Threshold Optimization
   - Phase 6: Final Test Evaluation & Business Impact

## Project Structure

```
orange-telecom-churn-prediction/
│
├── data/                          # Dataset files
│   ├── churn-bigml-80.csv        # Original 80% training set
│   ├── churn-bigml-20.csv        # Original 20% test set
│   └── README.md                 # Data documentation
│
├── images/                        # Plots and visualizations
│
├── orange_churn_analysis.ipynb   # Main Jupyter notebook with complete analysis
├── requirements.txt               # Python package dependencies
├── README.md                      # Project documentation (this file)
├── .gitignore                     # Git ignore rules
└── .gitattributes                 # Git attributes configuration
```

## Methodology

This project follows gold standard machine learning practices to prevent data leakage and ensure robust model evaluation:

### 1. Data Loading & Preprocessing
- Combined original train/test files and applied custom 80/20 stratified split
- Removed duplicate records globally before splitting
- Checked for missing values and data quality issues
- Maintained strict test set isolation (touched only once)

### 2. Exploratory Data Analysis (EDA)
- Performed on **training data only** to prevent information leakage
- Analyzed feature distributions, correlations, and churn patterns
- Identified key features: Account length, International plan, Customer service calls, Total charges

### 3. Feature Engineering & Pipeline Construction
- **Custom Transformers**: ColumnDropper, TargetEncoder, BinaryEncoder, Log1pTransformer
- **Pipeline Architecture**: Prevents cross-validation data leakage
- **Preprocessing Steps**: Encoding categorical variables, log transformations, standardization
- All transformers fit on training data only

### 4. Model Development & Comparison
Evaluated multiple algorithms using 5-fold stratified cross-validation on training data:
- **DummyClassifier**: Baseline sanity check (most frequent strategy)
- **Logistic Regression**: Linear baseline with balanced class weights
- **Random Forest**: Ensemble with balanced class weights
- **Gradient Boosting**: Sequential ensemble
- **XGBoost**: Best performing model

All models use `class_weight='balanced'` or equivalent to handle the imbalanced dataset (85% no churn / 15% churn).

### 5. Model Calibration (Training Data Only)
- Applied **CalibratedClassifierCV** with Platt Scaling (sigmoid method)
- Improves probability estimates for better business decision-making
- Calibrated on training data using cross-validation

### 6. Threshold Optimization (Training Data Only)
- Analyzed precision-recall tradeoff using cross-validated predictions on training data
- Selected optimal threshold targeting **80% recall** to maximize churn detection
- Business constraint: Willing to accept lower precision to catch more churners

### 7. Final Test Evaluation (Test Data - First Time)
- Evaluated calibrated model with optimized threshold on held-out test set
- **Test set touched exactly once** (gold standard practice)
- Comprehensive metrics: Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC, Brier Score

### 8. Business Impact Analysis
- ROI calculations for retention campaigns
- Generated actionable list of high-risk customers
- Cost-benefit analysis of intervention strategies

## Results

### Model Performance Comparison (5-Fold CV on Training Data)

| Model | Recall | Precision | F1-Score | ROC-AUC |
|-------|--------|-----------|----------|---------|
| DummyClassifier | 0.00% | 0.00% | 0.00% | 50.0% |
| Logistic Regression | ~75% | ~60% | ~67% | ~85% |
| Random Forest | ~70% | ~65% | ~67% | ~87% |
| Gradient Boosting | ~72% | ~68% | ~70% | ~89% |
| **XGBoost (Best)** | **~75%** | **~70%** | **~72%** | **~90%** |

*Note: Exact values in notebook. XGBoost selected as best model based on cross-validation performance.*

### Final Test Set Performance (After Calibration & Threshold Tuning)

**Key Metrics:**
- **Recall**: Target of 80% achieved (maximizes churn detection)
- **Precision**: Optimized based on business constraints
- **ROC-AUC**: Evaluates model's ranking ability
- **PR-AUC**: More informative for imbalanced datasets
- **Brier Score**: Measures calibration quality

**Business Impact:**
- Identified high-risk customers for targeted retention campaigns
- Cost-benefit analysis demonstrates positive ROI of intervention
- Actionable insights for customer service improvements

*Detailed results and visualizations available in the Jupyter notebook.*

## Acknowledgments

- Dataset provided by Orange Telecom via Kaggle
- Original data from KDD Cup 2009 competition
- Kaggle community for insights and best practices
- scikit-learn and XGBoost development teams for excellent ML libraries