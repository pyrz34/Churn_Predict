# 📊 Telco Customer Churn Prediction

Predicting customer churn is crucial for telecom companies to retain clients and improve business strategies. This project leverages machine learning models to predict which customers are likely to leave (churn).

---

## 🔹 Dataset
- **Source:** [Telco Customer Churn Dataset (Kaggle)](https://www.kaggle.com/blastchar/telco-customer-churn)
- **Observations:** 7043
- **Features:** 20 (demographics, subscription info, service usage)
- **Target Variable:** `Churn` (0 = stays, 1 = churns)

---

## 🛠️ Libraries & Tools
- Python 3.x
- Pandas, NumPy
- Scikit-learn (preprocessing, model selection, metrics)
- XGBoost, LightGBM
- Matplotlib, Seaborn (optional for visualization)

---

## ⚡ Project Workflow

### 1️⃣ Data Cleaning
- Corrected incorrect data types (e.g., `TotalCharges`)
- Filled missing values
- Detected and capped outliers

### 2️⃣ Feature Engineering
- Created `tenure_group` and `AvgCharges` columns

### 3️⃣ Encoding
- Binary features → Label Encoding
- Multi-class categorical features → One-Hot Encoding

### 4️⃣ Scaling
- Standardized all numerical features using `StandardScaler`

### 5️⃣ Modeling
- **Base Models:** Logistic Regression, KNN, SVC, Decision Tree, Random Forest, AdaBoost, Gradient Boosting, XGBoost, LightGBM
- **Hyperparameter Optimization:** GridSearchCV
- **Ensemble Model:** Soft Voting Classifier

### 6️⃣ Evaluation
- Metrics: Accuracy, F1 Score, ROC_AUC
- Confusion matrix to visualize performance

---

## 📈 Results
- After hyperparameter tuning, the best-performing models were: KNN, Random Forest, LightGBM  
- The ensemble Voting Classifier achieved high accuracy and AUC scores, making it reliable for churn prediction

---

## 🚀 How to Use
```python
# Preprocess the dataset
df_processed = preprocess_data(df)
X = df_processed.drop("Churn", axis=1)
y = df_processed["Churn"]

# Base models evaluation
base_models(X, y)

# Hyperparameter optimization
best_models = hyperparameter_optimization(X, y)

# Ensemble Voting Classifier
voting_clf = voting_classifier(best_models, X, y)
