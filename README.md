# 💳 Credit Card Fraud Detection using Machine Learning & Deep Learning

## 📌 Project Overview

This project focuses on detecting fraudulent credit card transactions using
a variety of Machine Learning (ML) classifiers and a Deep Learning model
built with Keras/TensorFlow. Given the highly imbalanced nature of fraud
datasets, special techniques like **Random Under-Sampling** and
**SMOTE (Synthetic Minority Over-sampling Technique)** were employed to
improve model performance and fairness.

---

## 📂 Dataset

Dataset : https://www.kaggle.com/datasets/siddheshasati/creditcard
- **Source File:** `creditcard.csv`
- **Total Records:** 284,807 transactions
- **Total Features:** 31 columns
  - `Time`, `V1` to `V28` (PCA-transformed features), `Amount`, `Class`
- **Target Variable:** `Class`
  - `0` → No Fraud (**99.83%** of dataset)
  - `1` → Fraud (**0.17%** of dataset)
- **Missing Values:** None (`df.isnull().sum().max() = 0`)
- **Data Types:** 30 float64 columns + 1 int64 column (`Class`)
- **Memory Usage:** 67.4 MB

> ⚠️ The dataset is **highly imbalanced**, which is a major challenge
> addressed throughout this project.

---

## 🛠️ Technologies & Libraries Used

| Category              | Libraries / Tools                                      |
|-----------------------|--------------------------------------------------------|
| Data Manipulation     | `numpy`, `pandas`                                      |
| Visualization         | `matplotlib`, `seaborn`                                |
| ML Algorithms         | `scikit-learn`                                         |
| Imbalance Handling    | `imbalanced-learn` (imblearn)                          |
| Deep Learning         | `tensorflow`, `keras`                                  |
| Dimensionality Reduc. | `TSNE`, `PCA`, `TruncatedSVD`                          |

---

## 🔄 What We Did — Step-by-Step

### 1. 📊 Exploratory Data Analysis (EDA)
- Loaded and inspected the dataset using `df.head()`, `df.info()`,
  `df.describe()`
- Confirmed **no null values** in the entire dataset
- Visualized **class distribution** — confirmed severe imbalance
  (99.83% No Fraud vs 0.17% Fraud)
- Plotted distribution of `Transaction Amount` and `Transaction Time`

### 2. 🧹 Data Preprocessing
- Applied **RobustScaler** (less prone to outliers) to scale the
  `Amount` and `Time` columns
- Replaced original `Amount` and `Time` with `scaled_amount` and
  `scaled_time`
- Rearranged columns to place scaled features at the front

### 3. ⚖️ Handling Class Imbalance

#### a. Random Under-Sampling
- Created a balanced subsample with equal fraud and non-fraud records
  (492 each)
- Used this subsample for correlation analysis and initial model training

#### b. SMOTE (Over-Sampling)
- Applied SMOTE on training data after train-test split to avoid
  data leakage
- Generated synthetic fraud samples to balance the training set

### 4. 🔍 Correlation & Feature Analysis
- Generated **Imbalanced Correlation Matrix** vs **SubSample
  Correlation Matrix**
- Identified features with **negative correlation** to fraud:
  `V17`, `V14`, `V12`, `V10`
- Identified features with **positive correlation** to fraud:
  `V11`, `V4`, `V2`, `V19`
- Visualized distributions of key fraud features (`V14`, `V12`, `V10`)

### 5. 🧹 Outlier Removal
Used **IQR (Interquartile Range)** method to remove extreme outliers
from fraud class for key features:

| Feature | Outliers Removed | Remaining Instances |
|---------|-----------------|---------------------|
| V14     | 4               | —                   |
| V12     | 4               | 976                 |
| V10     | 27              | 946                 |

### 6. 📉 Dimensionality Reduction (Visualization)
Applied three dimensionality reduction techniques to visualize
fraud vs non-fraud clusters:

| Technique      | Time Taken  |
|----------------|-------------|
| t-SNE          | ~9.6 s      |
| PCA            | ~0.012 s    |
| Truncated SVD  | ~0.005 s    |

> t-SNE produced the clearest cluster separation between fraud and
> non-fraud transactions.

### 7. 🤖 Model Training & Evaluation

#### ML Classifiers Used:
| Classifier                  | Training Accuracy | CV Score  | ROC-AUC Score |
|-----------------------------|-------------------|-----------|---------------|
| Logistic Regression         | 94.0%             | 94.58%    | **0.9786**    |
| K-Nearest Neighbors (KNN)   | 94.0%             | 94.05%    | 0.9393        |
| Support Vector Classifier   | 94.0%             | 94.31%    | 0.9759        |
| Decision Tree Classifier    | 93.0%             | 92.99%    | 0.9289        |

- Used **GridSearchCV** for hyperparameter tuning of all 4 classifiers
- Used **StratifiedKFold (5-fold)** cross-validation

#### Best Logistic Regression Parameters (via RandomizedSearchCV):
LogisticRegression(C=0.001, penalty='l1', random_state=42, solver='liblinear')



#### Sampling Technique Comparison:
| Technique                 | Accuracy Score |
|---------------------------|----------------|
| Random UnderSampling      | 0.9263         |
| **Oversampling (SMOTE)**  | **0.9759**     |

### 8. 🧠 Deep Learning Model (Keras / TensorFlow)

Two neural network models were built:

#### Under-Sampling Model Architecture:
| Layer   | Output Shape | Params |
|---------|-------------|--------|
| Dense   | (None, 30)  | 930    |
| Dense   | (None, 32)  | 992    |
| Dense   | (None, 2)   | 66     |
| **Total** | —         | **1,988** |

- **Optimizer:** Adam (lr=0.001)
- **Loss:** sparse_categorical_crossentropy
- **Epochs:** 20 | **Batch Size:** 25
- **Final Validation Accuracy:** ~96.05%

#### Over-Sampling (SMOTE) Model:
- **Epochs:** 20 | **Batch Size:** 300
- **Final Validation Accuracy:** ~99.99%
- Significantly better performance with SMOTE-balanced data

### 9. 📈 Model Performance Metrics

#### Classification Report — Logistic Regression (SMOTE):
| Class     | Precision | Recall | F1-Score | Support |
|-----------|-----------|--------|----------|---------|
| No Fraud  | 1.00      | 0.98   | 0.99     | 56,863  |
| Fraud     | 0.06      | 0.87   | 0.11     | 98      |
| Accuracy  |           |        | **0.98** | 56,961  |

#### Classification Report — All Classifiers (Under-Sampling test set):
| Classifier              | Accuracy |
|-------------------------|----------|
| Logistic Regression     | **0.93** |
| KNears Neighbors        | **0.93** |
| Support Vector Classifier | 0.92   |
| Decision Tree Classifier  | 0.90   |

#### Precision-Recall:
- **UnderSampling** Average Precision-Recall Score: **0.06**
- **OverSampling (SMOTE)** Average Precision-Recall Score: **0.69**

---

## 🏆 Outcome & Key Findings

1. **Logistic Regression** achieved the highest ROC-AUC score of
   **0.9786** among all ML classifiers.
2. **SMOTE (OverSampling)** significantly outperformed Random
   UnderSampling, improving accuracy from **0.926** to **0.976**.
3. The Deep Learning model trained on SMOTE data achieved near-perfect
   validation accuracy of **~99.99%** by epoch 20.
4. **Precision-Recall** analysis revealed that SMOTE dramatically
   improved the average precision-recall score from **0.06** (under-
   sampling) to **0.69** (oversampling).
5. Key fraud-correlated features identified: `V14`, `V12`, `V10`
   (negative) and `V11`, `V4`, `V2`, `V19` (positive).
6. Accuracy alone is **misleading** for highly imbalanced fraud datasets
   — F1-Score, ROC-AUC, and Precision-Recall curves are more reliable.

---

## 📁 Project Structure

credit-card-fraud-detection/ │ ├── creditcard.csv # Dataset ├── fraud_detection.ipynb # Main Jupyter Notebook ├── README.md # Project Documentation └── requirements.txt # Dependencies







