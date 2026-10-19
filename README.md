# 💳 Credit Card Fraud Detection

An end-to-end Machine Learning project designed to detect potentially fraudulent credit card transactions using imbalanced classification techniques and an interactive Streamlit web application.

## 📌 Project Overview

Credit card fraud detection is a challenging classification problem because fraudulent transactions represent only a small fraction of all transactions.

This project covers the complete machine learning workflow, including data exploration, preprocessing, model training, evaluation, prediction, automated testing, and an interactive application.

## 🎯 Objectives

- Analyze credit card transaction data.
- Explore patterns in fraudulent and legitimate transactions.
- Handle class imbalance using class weighting and SMOTE.
- Train and compare Logistic Regression models.
- Evaluate models using appropriate classification metrics.
- Build a reusable prediction pipeline.
- Create an interactive Streamlit application.
- Test prediction functionality using pytest.

## 📂 Dataset

The dataset was downloaded from Kaggle.

### Features

| Column | Description |
|---|---|
| `Time` | Seconds elapsed since the first transaction |
| `V1`–`V28` | Anonymized transaction features |
| `Amount` | Transaction amount |
| `Class` | Target variable: 0 = legitimate, 1 = fraud |

The dataset contains 30 columns, including 28 anonymized features, transaction time, transaction amount, and the target label.

**Note:** Refer to the original Kaggle dataset page for its license and attribution requirements.

## 🛠️ Technologies Used

- Python
- Pandas and NumPy
- Scikit-learn
- Imbalanced-learn (SMOTE)
- Matplotlib and Seaborn
- Joblib
- Streamlit
- Pytest
- Jupyter Notebook
- Git and GitHub

## 🤖 Machine Learning Models

Two Logistic Regression approaches were evaluated:

1. **Weighted Logistic Regression** — uses class weights to account for class imbalance.
2. **SMOTE Logistic Regression** — uses Synthetic Minority Over-sampling Technique to oversample the minority class during training.

## 📊 Model Evaluation Results

| Metric | Weighted Logistic Regression | SMOTE Logistic Regression |
|---|---:|---:|
| Precision | 0.0610 | 0.0578 |
| Recall | 0.9184 | 0.9184 |
| F1-score | 0.1144 | 0.1088 |
| Average Precision | 0.7190 | 0.7245 |
| ROC-AUC | 0.9721 | 0.9708 |

### Results Interpretation

- Both models achieved approximately **91.84% recall**, detecting most fraudulent transactions in the evaluated test set.
- Precision was low, meaning many transactions flagged as fraud may actually be legitimate.
- SMOTE Logistic Regression achieved slightly higher Average Precision.
- Weighted Logistic Regression achieved slightly higher ROC-AUC and F1-score.

Model selection should consider false-positive costs, the precision-recall trade-off, and the decision threshold, rather than relying on accuracy alone.

*These are the recorded evaluation results. Final comparisons should use the same held-out test set and evaluation procedure.*

## 📁 Project Structure

```text
fraud_detection/
├── app/
│   ├── __init__.py
│   └── streamlit_app.py
├── data/
│   └── raw/
├── dashboard/
├── models/
├── notebooks/
│   ├── data/
│   └── EDA.ipynb
├── src/
│   ├── __init__.py
│   └── predict.py
├── tests/
│   └── test_predict.py
├── .gitignore
├── main.py
├── requirements.txt
└── README.md
```

*Adjust the structure above if your actual folders or filenames differ.*

## ⚙️ Installation and Setup

### 1. Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd fraud_detection
```

Replace `YOUR_GITHUB_REPOSITORY_URL` with your actual repository URL.

### 2. Create a virtual environment

On Windows PowerShell:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```powershell
python -m pip install -r requirements.txt
```

### 4. Verify dependencies

```powershell
python -m pip check
```

## 🚀 Run the Streamlit Application

From the project root, run:

```powershell
python -m streamlit run app/streamlit_app.py
```

Streamlit will provide a local URL, usually:

```text
http://localhost:8501
```

Ensure that the trained model artifact exists at the path expected by the prediction code before starting the application.

## 🧪 Run Tests

Execute the prediction pipeline tests:

```powershell
python -m pytest tests/test_predict.py -v
```

The current test suite contains eight tests covering feature validation, feature ordering, invalid input, prediction outputs, threshold classification, and empty input handling.

## 📈 Key Learnings

- Handling highly imbalanced datasets.
- Understanding precision, recall, F1-score, Average Precision, and ROC-AUC.
- Comparing class weighting with SMOTE.
- Building reusable model prediction functions.
- Validating inputs and testing application logic.
- Connecting a machine learning model to a Streamlit application.
- Organizing a data science project into reusable modules.

## 🔮 Future Improvements

- Evaluate additional models such as Random Forest and gradient boosting.
- Optimize the classification threshold using business costs.
- Add confusion matrix and precision-recall curve visualizations.
- Improve input validation and error handling.
- Add automated testing to a CI workflow.
- Deploy the application to a hosting platform.
- Monitor model performance and potential data drift.

## ⚠️ Limitations

- High recall does not guarantee high precision.
- False-positive alerts may create additional investigation costs.
- Model results depend on the dataset and evaluation methodology.
- Anonymized features limit direct interpretation.
- The application is an educational prototype, not a production-grade fraud prevention system.

## ⚖️ Disclaimer

This project is intended for educational and portfolio purposes only. Predictions should not be treated as definitive evidence of fraud or used as the sole basis for financial decisions.

## 👩‍💻 Author

**Divyata Maurya**

GitHub: 