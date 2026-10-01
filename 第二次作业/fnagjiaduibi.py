import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import ElasticNetCV, LassoCV, RidgeCV
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 1. 载入数据并划分训练集与测试集
data = fetch_california_housing(as_frame=True)
X, y = data.data, data.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 2. 特征 Z-score 标准化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 3. 分别使用 RidgeCV、LassoCV、ElasticNetCV 建模 (cv=10)
# RidgeCV
ridge_cv = RidgeCV(alphas=np.logspace(-3, 3, 100), cv=10)
ridge_cv.fit(X_train_scaled, y_train)
ridge_pred = ridge_cv.predict(X_test_scaled)
ridge_rmse = np.sqrt(mean_squared_error(y_test, ridge_pred))

# LassoCV
lasso_cv = LassoCV(alphas=np.logspace(-3, 3, 100), cv=10, random_state=42)
lasso_cv.fit(X_train_scaled, y_train)
lasso_pred = lasso_cv.predict(X_test_scaled)
lasso_rmse = np.sqrt(mean_squared_error(y_test, lasso_pred))
lasso_non_zero = np.sum(lasso_cv.coef_ != 0)

# ElasticNetCV
elastic_cv = ElasticNetCV(
    alphas=np.logspace(-3, 3, 100),
    l1_ratio=[0.1, 0.3, 0.5, 0.7, 0.9],
    cv=10,
    random_state=42,
)
elastic_cv.fit(X_train_scaled, y_train)
elastic_pred = elastic_cv.predict(X_test_scaled)
elastic_rmse = np.sqrt(mean_squared_error(y_test, elastic_pred))
elastic_non_zero = np.sum(elastic_cv.coef_ != 0)

# 4. 报告评估结果
print(f"--- 建模结果对比 ---")
print(f"Ridge 最佳 alpha: {ridge_cv.alpha_:.4f} | 测试集 RMSE: {ridge_rmse:.4f}")
print(
    f"Lasso 最佳 alpha: {lasso_cv.alpha_:.4f} | 测试集 RMSE: {lasso_rmse:.4f}"
    f" | 非零变量数: {lasso_non_zero}"
)
print(
    f"ElasticNet 最佳 alpha: {elastic_cv.alpha_:.4f}, l1_ratio:"
    f" {elastic_cv.l1_ratio_:.2f} | 测试集 RMSE: {elastic_rmse:.4f} | 非零变量数:"
    f" {elastic_non_zero}"
)