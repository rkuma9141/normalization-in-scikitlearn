# 🔢 Normalization using MinMaxScaler

This project demonstrates how to apply **Normalization** (MinMax Scaling) to a dataset using **scikit-learn** in Python.

---

## 📘 What is Normalization?

**Normalization** is a feature scaling technique that rescales the values of numerical columns into a fixed range, typically **0 to 1**.

### ✅ Formula:
\[
X_{scaled} = \frac{X - X_{min}}{X_{max} - X_{min}}
\]

---

## 🎯 Why Normalize?

- Different features may have different scales (e.g., Age vs. Salary)
- Many ML algorithms like **KNN**, **SVM**, and **Gradient Descent** perform better on normalized data
- Helps improve **model performance** and **training speed**

---

## ⚙️ Steps Performed:

1. Loaded a sample dataset using pandas
2. Split the dataset into input features (X) and target variable (y)
3. Applied `train_test_split`
4. Applied `MinMaxScaler` only on training data
5. Transformed both training and test features
6. Combined scaled features with target variable (optional)

---

## 🧪 Sample Code:

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# Step 1: Load dataset
data = {
    'Age': [18, 25, 35, 45, 55],
    'Salary': [15000, 25000, 35000, 45000, 55000],
    'Purchased': [0, 1, 0, 1, 0]
}
df = pd.DataFrame(data)

# Step 2: Split into features and target
X = df[['Age', 'Salary']]
y = df['Purchased']

# Step 3: Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Apply MinMaxScaler on training data
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 5: Convert back to DataFrame
X_train_scaled = pd.DataFrame(X_train_scaled, columns=X.columns)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=X.columns)
