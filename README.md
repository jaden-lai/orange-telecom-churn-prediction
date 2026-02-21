# Orange Telecom Churn Prediction

A machine learning project to predict customer churn for Orange Telecom using classification algorithms.

## 📊 Project Overview

Customer churn is a critical metric for telecom companies, and many big company's today. This project analyzes customer data to predict which customers are likely to leave and understand user behaviour.

I decided to use this dataset on kaggle, https://www.kaggle.com/datasets/mnassrib/telecom-churn-datasets?resource=download, a cleaned version of https://www.kdd.org/kdd-cup/view/kdd-cup-2009, which was uses "databases from the French Telecom company Orange to predict the propensity of customers to switch provider".

## 🎯 Objectives

- Perform exploratory data analysis (EDA) on telecom customer data
- Identify key factors contributing to customer churn
- Build and compare multiple classification models
- Provide actionable insights for customer retention

## 📁 Dataset

- **Source**: [Orange Telecom Customer Churn Dataset (Kaggle)](https://www.kaggle.com/datasets/mnassrib/telecom-churn-datasets)
- **Original Source**: [KDD Cup 2009](https://www.kdd.org/kdd-cup/view/kdd-cup-2009) - French Telecom company Orange
- **Size**: ~3,333 customer records (combined from provided train/test files)
- **Features**: 20 variables including customer demographics, usage patterns, and service information
- **Target**: Binary classification (Churn: True/False)
- **Split Strategy**: Custom 80/20 train/test split with stratification (random_state=42)

## 🔧 Technologies Used

- **Python 3.x**
- **Libraries**: 
  - Data Analysis: `pandas`, `numpy`
  - Visualization: `matplotlib`, `seaborn`
  - Machine Learning: `scikit-learn`
  - (Add more as you use them)

## 📂 Project Structure

```
├── data/                          # Dataset files
│   ├── churn-bigml-80.csv        # Original training set
│   └── churn-bigml-20.csv        # Original test set
├── images/                        # Visualizations and plots
├── orange_churn_analysis.ipynb   # Main analysis notebook
├── README.md                      # Project documentation
├── requirements.txt               # Python dependencies
└── .gitignore                     # Git ignore file
```

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- Jupyter Notebook

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/orange_telecom_churn_prediction.git
cd orange_telecom_churn_prediction
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

3. Launch Jupyter Notebook:
```bash
jupyter notebook
```

4. Open `orange_churn_analysis.ipynb` to see the analysis

## 📈 Key Findings

<!-- Update this section after completing your analysis -->

- Finding 1: [e.g., Feature X is the strongest predictor of churn]
- Finding 2: [e.g., Model Y achieved 85% accuracy]
- Finding 3: [e.g., Top 3 churn indicators are...]

## 🧪 Methodology

1. **Data Loading & Splitting**
   - Combine source datasets
   - Custom 80/20 train/test split with stratification
   - Maintain consistent random state (42) for reproducibility

2. **Data Cleaning & Preprocessing**
   - Check for missing values and duplicates
   - Handle categorical variables
   - Feature engineering and transformation

3. **Exploratory Data Analysis**
   - Univariate and bivariate analysis
   - Correlation analysis
   - Visualization of key patterns and churn indicators

4. **Model Development**
   - Baseline model establishment
   - Multiple algorithm comparison
   - Hyperparameter tuning

5. **Evaluation**
   - Performance metrics (Accuracy, Precision, Recall, F1-Score)
   - ROC-AUC analysis
   - Model comparison

## 📊 Results

<!-- Add model performance metrics here -->

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| Model 1 | XX% | XX% | XX% | XX% |
| Model 2 | XX% | XX% | XX% | XX% |

## 💡 Future Improvements

- [ ] Implement additional algorithms (XGBoost, Random Forest, Neural Networks)
- [ ] Perform feature selection to identify most important variables
- [ ] Deploy model as a web application
- [ ] Create interactive dashboard for predictions

## 👤 Author

**Your Name**
- LinkedIn: [Your LinkedIn](https://linkedin.com/in/yourprofile)
- Portfolio: [Your Website](https://yourwebsite.com)
- Email: your.email@example.com

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Dataset provided by Orange Telecom
- Kaggle community for insights and best practices

---

⭐ If you found this project helpful, please consider giving it a star!
