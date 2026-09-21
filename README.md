# EX-NO.-4a-MACHINE-LEARNING-MODEL-LINEAR-REGRESSION
# MACHINE LEARNING MODEL: REGRESSION

## AIM

To predict house prices using different regression models and compare their performance using standard regression evaluation metrics.

---

## PROCEDURE

1. Import the required Python libraries for data handling, visualization, preprocessing, and machine learning.
2. Load the House Price Dataset using Pandas.
3. Display the dataset and examine its information, shape, descriptive statistics, and missing values.
4. Perform Exploratory Data Analysis (EDA) using histograms, scatter plots, correlation analysis, and boxplots.
5. Detect and remove extreme house-price values using the 1st and 99th percentiles.
6. Select the input features:
   - `square_feet`
   - `num_rooms`
   - `age`
   - `distance_to_city(km)`
7. Select `price` as the target variable.
8. Split the dataset into training and testing sets using an 80:20 ratio.
9. Apply `StandardScaler` to scale the input features.
10. Train the following regression models:
    - Linear Regression
    - Ridge Regression
    - Lasso Regression
    - ElasticNet Regression
    - Polynomial Regression
    - Decision Tree Regressor
    - Random Forest Regressor
    - Gradient Boosting Regressor
    - Support Vector Regressor (SVR)
    - K-Nearest Neighbors (KNN) Regressor
11. Generate predictions using each trained model.
12. Evaluate the models using:
    - RMSE (Root Mean Squared Error)
    - MAE (Mean Absolute Error)
    - R² Score
13. Compare the performance of all regression models.
14. Analyze the actual and predicted house prices.
15. Compare the models based on their evaluation metrics.

---

# THEORY

## 1. Linear Regression

Linear Regression is a supervised learning algorithm used to predict continuous numerical values.

It finds a linear relationship between the input features and the target variable.

In this experiment, Linear Regression is used to predict house prices from features such as square feet, number of rooms, age, and distance to the city.

---

## 2. Ridge Regression

Ridge Regression is a regularized form of Linear Regression.

It uses L2 regularization to control the size of the model coefficients.

It helps reduce the effect of large coefficients and can help prevent overfitting.

In this experiment, Ridge Regression is used for house-price prediction.

---

## 3. Lasso Regression

Lasso Regression is a Linear Regression technique that uses L1 regularization.

L1 regularization can reduce some feature coefficients toward zero.

This can also help identify less important features.

In this experiment, Lasso Regression is used to predict house prices.

---

## 4. ElasticNet Regression

ElasticNet Regression combines L1 and L2 regularization.

It combines the properties of Lasso and Ridge Regression.

ElasticNet is useful when multiple input features contribute to the prediction.

In this experiment, ElasticNet is used for house-price prediction.

---

## 5. Polynomial Regression

Polynomial Regression extends Linear Regression by creating polynomial features.

It can model nonlinear relationships between the input features and the target variable.

In this experiment, polynomial features with degree 2 are generated and Linear Regression is applied to them.

---

## 6. Decision Tree Regressor

A Decision Tree Regressor predicts continuous values by dividing the data into different regions based on feature values.

It makes a sequence of decisions using the input features and produces a prediction at the final leaf node.

Decision Trees can model nonlinear relationships between input features and the target.

---

## 7. Random Forest Regressor

Random Forest Regressor is an ensemble learning method that combines multiple decision trees.

Each tree produces a prediction, and the predictions are combined to obtain the final result.

Using multiple trees can provide more robust predictions than a single decision tree.

---

## 8. Gradient Boosting Regressor

Gradient Boosting Regressor is an ensemble learning technique that builds models sequentially.

Each new model attempts to reduce the errors made by the previous models.

The individual models are combined to produce the final prediction.

---

## 9. Support Vector Regressor (SVR)

Support Vector Regressor is a regression technique based on Support Vector Machine principles.

It attempts to find a function that predicts target values while keeping prediction errors within an acceptable margin.

The model used in this experiment uses an RBF kernel.

SVR can model nonlinear relationships between the input features and target variable.

---

## 10. K-Nearest Neighbors (KNN) Regressor

K-Nearest Neighbors Regressor predicts a target value based on nearby observations in the feature space.

For a new input, KNN identifies the nearest training observations and uses their target values to make the prediction.

In this experiment, 5 nearest neighbors are used.

---

# PROGRAM

## 1. Linear Regression

```python
lr = LinearRegression()

lr.fit(X_train_scaled, y_train)

y_pred_lr = lr.predict(X_test_scaled)
```

## 2. Ridge Regression

```python
ridge = Ridge(alpha=1.0)

ridge.fit(X_train_scaled, y_train)

y_pred_ridge = ridge.predict(X_test_scaled)
```

## 3. Lasso Regression

```python
lasso = Lasso(alpha=0.1)

lasso.fit(X_train_scaled, y_train)

y_pred_lasso = lasso.predict(X_test_scaled)
```

## 4. ElasticNet Regression

```python
elastic = ElasticNet(
    alpha=0.1,
    l1_ratio=0.5
)

elastic.fit(X_train_scaled, y_train)

y_pred_elastic = elastic.predict(X_test_scaled)
```

## 5. Polynomial Regression

```python
poly = PolynomialFeatures(degree=2)

X_train_poly = poly.fit_transform(X_train_scaled)
X_test_poly = poly.transform(X_test_scaled)

poly_lr = LinearRegression()

poly_lr.fit(X_train_poly, y_train)

y_pred_poly = poly_lr.predict(X_test_poly)
```

## 6. Decision Tree Regressor

```python
dt = DecisionTreeRegressor(
    random_state=42
)

dt.fit(X_train_scaled, y_train)

y_pred_dt = dt.predict(X_test_scaled)
```

## 7. Random Forest Regressor

```python
rf = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

rf.fit(X_train_scaled, y_train)

y_pred_rf = rf.predict(X_test_scaled)
```

## 8. Gradient Boosting Regressor

```python
gbr = GradientBoostingRegressor(
    n_estimators=100,
    learning_rate=0.1,
    random_state=42
)

gbr.fit(X_train_scaled, y_train)

y_pred_gbr = gbr.predict(X_test_scaled)
```

## 9. Support Vector Regressor (SVR)

```python
svr = SVR(
    kernel='rbf',
    C=100,
    gamma=0.1,
    epsilon=.1
)

svr.fit(X_train_scaled, y_train)

y_pred_svr = svr.predict(X_test_scaled)
```

## 10. K-Nearest Neighbors (KNN) Regressor

```python
knn = KNeighborsRegressor(
    n_neighbors=5
)

knn.fit(X_train_scaled, y_train)

y_pred_knn = knn.predict(X_test_scaled)
```
## Program
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures

from sklearn.linear_model import (
    LinearRegression,
    Ridge,
    Lasso,
    ElasticNet
)

from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor
)

from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor

from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score
)

# Load Dataset
df = pd.read_csv(
    '/content/drive/MyDrive/Datasets/house_prices_dataset.csv'
)

# Display first rows
df.head()

# Display dataset
df

# Display information
df.info()

# Display shape
df.shape

# Summary statistics
df.describe()

# Check missing values
df.isnull().sum()

# Distribution of Features
numeric_features = [
    'square_feet',
    'num_rooms',
    'age',
    'distance_to_city(km)',
    'price'
]

for col in numeric_features:
    plt.figure(figsize=(6,4))
    sns.histplot(df[col], kde=True, bins=30)
    plt.title(f'Distribution of {col}')
    plt.show()

# Correlation Analysis
plt.figure(figsize=(8,6))

sns.heatmap(
    df.corr(),
    annot=True,
    cmap='coolwarm',
    fmt=".2f"
)

plt.title("Feature Correlation Matrix")
plt.show()

# Scatter Plots
for col in [
    'square_feet',
    'num_rooms',
    'age',
    'distance_to_city(km)'
]:
    plt.figure(figsize=(6,4))
    sns.scatterplot(x=df[col], y=df['price'])
    plt.title(f'{col} vs Price')
    plt.show()

# Outlier Detection
for col in [
    'square_feet',
    'num_rooms',
    'age',
    'distance_to_city(km)',
    'price'
]:
    plt.figure(figsize=(6,4))
    sns.boxplot(df[col])
    plt.title(f'Boxplot of {col}')
    plt.show()

# Outlier Treatment
Q1 = df['price'].quantile(0.01)
Q99 = df['price'].quantile(0.99)

df = df[
    (df['price'] >= Q1) &
    (df['price'] <= Q99)
]

# Define Features and Target
X = df[
    [
        'square_feet',
        'num_rooms',
        'age',
        'distance_to_city(km)'
    ]
]

y = df['price']

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Feature Scaling
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled = scaler.transform(X_test)

# Baseline Model
y_pred_baseline = (
    np.mean(y_train) *
    np.ones_like(y_test)
)

rmse_baseline = np.sqrt(
    mean_squared_error(y_test, y_pred_baseline)
)

mae_baseline = mean_absolute_error(
    y_test,
    y_pred_baseline
)

print(
    f"Baseline RMSE: {rmse_baseline:.2f}, "
    f"MAE: {mae_baseline:.2f}"
)

# Linear Regression
lr = LinearRegression()

lr.fit(X_train_scaled, y_train)

y_pred_lr = lr.predict(X_test_scaled)

# Ridge Regression
ridge = Ridge(alpha=1.0)

ridge.fit(X_train_scaled, y_train)

y_pred_ridge = ridge.predict(X_test_scaled)

# Lasso Regression
lasso = Lasso(alpha=0.1)

lasso.fit(X_train_scaled, y_train)

y_pred_lasso = lasso.predict(X_test_scaled)

# ElasticNet Regression
elastic = ElasticNet(
    alpha=0.1,
    l1_ratio=0.5
)

elastic.fit(X_train_scaled, y_train)

y_pred_elastic = elastic.predict(X_test_scaled)

# Polynomial Regression
poly = PolynomialFeatures(degree=2)

X_train_poly = poly.fit_transform(X_train_scaled)
X_test_poly = poly.transform(X_test_scaled)

poly_lr = LinearRegression()

poly_lr.fit(X_train_poly, y_train)

y_pred_poly = poly_lr.predict(X_test_poly)

# Decision Tree Regressor
dt = DecisionTreeRegressor(
    random_state=42
)

dt.fit(X_train_scaled, y_train)

y_pred_dt = dt.predict(X_test_scaled)

# Random Forest Regressor
rf = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

rf.fit(X_train_scaled, y_train)

y_pred_rf = rf.predict(X_test_scaled)

# Gradient Boosting Regressor
gbr = GradientBoostingRegressor(
    n_estimators=100,
    learning_rate=0.1,
    random_state=42
)

gbr.fit(X_train_scaled, y_train)

y_pred_gbr = gbr.predict(X_test_scaled)

# Support Vector Regressor
svr = SVR(
    kernel='rbf',
    C=100,
    gamma=0.1,
    epsilon=.1
)

svr.fit(X_train_scaled, y_train)

y_pred_svr = svr.predict(X_test_scaled)

# K-Nearest Neighbors Regressor
knn = KNeighborsRegressor(
    n_neighbors=5
)

knn.fit(X_train_scaled, y_train)

y_pred_knn = knn.predict(X_test_scaled)

# Model Evaluation
models = {
    "Linear Regression": y_pred_lr,
    "Ridge": y_pred_ridge,
    "Lasso": y_pred_lasso,
    "ElasticNet": y_pred_elastic,
    "Polynomial Regression": y_pred_poly,
    "Decision Tree": y_pred_dt,
    "Random Forest": y_pred_rf,
    "Gradient Boosting": y_pred_gbr,
    "SVR": y_pred_svr,
    "KNN": y_pred_knn
}

results = []

for name, y_pred in models.items():

    rmse = np.sqrt(
        mean_squared_error(y_test, y_pred)
    )

    mae = mean_absolute_error(
        y_test,
        y_pred
    )

    r2 = r2_score(
        y_test,
        y_pred
    )

    results.append([
        name,
        rmse,
        mae,
        r2
    ])

results_df = pd.DataFrame(
    results,
    columns=["Model", "RMSE", "MAE", "R2"]
)

results_df.sort_values(
    by="RMSE"
)

# Actual vs Predicted Price
plt.figure(figsize=(15,12))

for i, (name, y_pred) in enumerate(models.items()):

    plt.subplot(5,2,i+1)

    plt.scatter(
        y_test,
        y_pred,
        alpha=0.5
    )

    plt.plot(
        [y_test.min(), y_test.max()],
        [y_test.min(), y_test.max()],
        'r--'
    )

    plt.xlabel("Actual Price")
    plt.ylabel("Predicted Price")
    plt.title(f"{name}: Actual vs Predicted")

plt.tight_layout()
plt.show()

# Residual Analysis
for name, y_pred in models.items():

    residuals = y_test - y_pred

    plt.figure(figsize=(6,4))

    sns.scatterplot(
        x=y_pred,
        y=residuals,
        alpha=0.5
    )

    plt.axhline(
        0,
        color='r',
        linestyle='--'
    )

    plt.xlabel("Predicted Price")
    plt.ylabel("Residuals")
    plt.title(f"{name}: Residual Plot")

    plt.show()

# Random Forest Feature Importance
importances = rf.feature_importances_

feat_names = X.columns

plt.figure(figsize=(6,4))

sns.barplot(
    x=importances,
    y=feat_names
)

plt.title(
    "Random Forest Feature Importance"
)

plt.show()

# Gradient Boosting Feature Importance
importances_gbr = gbr.feature_importances_

plt.figure(figsize=(6,4))

sns.barplot(
    x=importances_gbr,
    y=feat_names
)

plt.title("Gradient Boosting Feature Importance")

plt.show()

# RMSE Comparison Graph
plt.figure(figsize=(10,6))

sns.barplot(
    x="RMSE",
    y="Model",
    data=results_df.sort_values("RMSE")
)

plt.title(
    "RMSE Comparison Across Regression Models"
)

plt.show()
```
## Output
<img width="1731" height="530" alt="image" src="https://github.com/user-attachments/assets/566544b0-1afd-41b7-9fce-bfa20fc3f446" />
<img width="1666" height="810" alt="image" src="https://github.com/user-attachments/assets/3fb94b97-5e46-425b-af6b-7d57f16c2c9b" />
<img width="848" height="933" alt="image" src="https://github.com/user-attachments/assets/fb5d40c6-1396-4902-9f7d-096c9ae370b0" />
<img width="1225" height="689" alt="image" src="https://github.com/user-attachments/assets/1f72b65a-3aca-4444-9425-8c1a8e0984cc" />


LINK FOR FULL SCREEEN OUTPUT:https://colab.research.google.com/drive/1PG9UiFnmUwdmBNDAZIXosSs33yyEf3JH#scrollTo=kZDJ77xmL2PB&fullscreenOutput=true

# CONCLUSION

Thus, different regression models were applied for house-price prediction using the selected house-related features.

The models included Linear Regression, Ridge Regression, Lasso Regression, ElasticNet Regression, Polynomial Regression, Decision Tree Regressor, Random Forest Regressor, Gradient Boosting Regressor, Support Vector Regressor, and K-Nearest Neighbors Regressor.

The performance of the models can be compared using RMSE, MAE, and R² score. Lower RMSE and MAE indicate smaller prediction errors, while a higher R² score generally indicates better explanation of the variation in house prices.

The experiment demonstrates the application and comparison of different regression techniques for continuous-value prediction.evaluation metrics.

