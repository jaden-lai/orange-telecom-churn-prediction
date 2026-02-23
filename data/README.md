# Data Directory

This directory contains the datasets used for the Orange Telecom Churn Prediction project.

## How to Obtain the Dataset

**The dataset files are not included in this repository.** You must download them from Kaggle:

1. Visit the [Orange Telecom Churn Dataset on Kaggle](https://www.kaggle.com/datasets/mnassrib/telecom-churn-datasets)
2. Click the "Download" button (requires free Kaggle account login)
3. Extract the downloaded ZIP file
4. Place the following CSV files in this `data/` directory:
   - `churn-bigml-80.csv`
   - `churn-bigml-20.csv`

**Note**: These files should be placed directly in the `data/` folder alongside this README file.

## Dataset Files

### 1. churn-bigml-80.csv
- **Purpose**: Training dataset (80% split)
- **Size**: ~2,667 records
- **Description**: Used for model training and exploratory data analysis

### 2. churn-bigml-20.csv
- **Purpose**: Test dataset (20% split)
- **Size**: ~666 records
- **Description**: Used for final model evaluation

## Dataset Features

The dataset contains **20 features** including:

### Categorical Features
- `State`: Customer's state (US state code)
- `Area code`: Telephone area code
- `International plan`: Whether customer has international plan (Yes/No)
- `Voice mail plan`: Whether customer has voice mail plan (Yes/No)

### Numerical Features
- `Account length`: Number of days the account has been active
- `Number vmail messages`: Number of voice mail messages
- `Total day minutes`: Total minutes of day calls
- `Total day calls`: Total number of day calls
- `Total day charge`: Total charge for day calls
- `Total eve minutes`: Total minutes of evening calls
- `Total eve calls`: Total number of evening calls
- `Total eve charge`: Total charge for evening calls
- `Total night minutes`: Total minutes of night calls
- `Total night calls`: Total number of night calls
- `Total night charge`: Total charge for night calls
- `Total intl minutes`: Total minutes of international calls
- `Total intl calls`: Total number of international calls
- `Total intl charge`: Total charge for international calls
- `Customer service calls`: Number of calls to customer service

### Target Variable
- `Churn`: Whether the customer churned (True/False)

## Data Source

This dataset is a standard telecom churn dataset used for classification tasks and is publicly available for educational and portfolio projects.

## Usage

The datasets are combined and split custom in the analysis notebook:

```python
from sklearn.model_selection import train_test_split

# Load both files
df1 = pd.read_csv('data/churn-bigml-80.csv')
df2 = pd.read_csv('data/churn-bigml-20.csv')

# Combine datasets
df_full = pd.concat([df1, df2], axis=0, ignore_index=True)

# Create custom train/test split with stratification
X = df_full.drop('Churn', axis=1)
y = df_full['Churn']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
```

This approach ensures:
- **Reproducibility**: Fixed random seed (42) for consistent splits
- **Class balance**: Stratification maintains churn rate in both sets
- **Control**: Custom split ratio and methodology

## Notes

- No missing values in the dataset
- Target variable is imbalanced (typical churn rate ~14-15%)
- All charges are in USD
- Time-based features (day/evening/night) represent different calling periods
