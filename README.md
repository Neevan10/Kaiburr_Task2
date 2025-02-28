# Kaiburr_Task2
Customer Complaint Text Classification
# Customer Complaint Text Classification  

## Overview  
This project focuses on **text classification** for consumer complaints. The goal is to categorize complaints into predefined categories using **machine learning models**.  

### Dataset  
The dataset is sourced from the [Consumer Complaint Database](https://catalog.data.gov/dataset/consumer-complaint-database). It includes various complaints related to financial services.  

**Categories:**  
0. Credit reporting, repair, or other  
1. Debt collection  
2. Consumer Loan  
3. Mortgage  

## Steps Involved  
1. **Exploratory Data Analysis (EDA) and Feature Engineering**  
2. **Text Preprocessing** (Cleaning and Tokenization)  
3. **Model Selection for Multi-class Classification**  
4. **Comparison of Model Performance**  
5. **Model Evaluation and Metrics Calculation**  
6. **Prediction on New Complaints**  

## Installation  

### Prerequisites  
Ensure that you have **Python 3.7+** installed.  

Install the required libraries using:  

```bash
pip install numpy pandas matplotlib seaborn nltk scikit-learn
```

## Implementation  

### 1. Importing Libraries  
The following libraries are used for data processing and model training:  

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
import string

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import confusion_matrix, classification_report
```

### 2. Load Dataset  
```python
df = pd.read_csv('complaints.csv')
df.head().T
```

### 3. Text Preprocessing  
- Removing **stopwords**, **punctuations**, and **special characters**  
- Applying **TF-IDF vectorization**  

```python
from sklearn.feature_extraction.text import TfidfTransformer, TfidfVectorizer

vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
X = vectorizer.fit_transform(df['complaint_text'])
y = df['category']
```

### 4. Splitting Data  
```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

### 5. Training Models  

#### Logistic Regression  
```python
lr = LogisticRegression()
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)
print(classification_report(y_test, y_pred))
```

#### Random Forest  
```python
rf = RandomForestClassifier(n_estimators=100)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
print(classification_report(y_test, y_pred_rf))
```

### 6. Model Evaluation  
The models are evaluated using:  
- **Confusion Matrix**  
- **Accuracy Score**  
- **F1-score**  

```python
from sklearn.metrics import confusion_matrix

conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt='d')
plt.show()
```

## Results  
The project compares different classifiers and selects the best-performing model based on evaluation metrics.

## How to Run  
1. Clone the repository  
```bash
git clone https://github.com/your-repo/Customer-Complaint-Classification.git
cd Customer-Complaint-Classification
```
2. Install dependencies  
```bash
pip install -r requirements.txt
```
3. Run the notebook in Jupyter  
```bash
jupyter notebook Customer\ Complain\ Text\ Classification.ipynb
```

