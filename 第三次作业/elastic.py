import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNetCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 自动生成模拟数据，不需要外部csv
np.random.seed(42)
n_samples = 1000
X = pd.DataFrame(np.random.randn(n_samples, 5), columns=["f1","f2","f3","f4","f5"])
y = 2 * X["f1"] + 3 * X["f3"] + np.random.randn(n_samples) * 0.4

# 划分训练集测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 标准化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 弹性网交叉验证
elastic_net_cv = ElasticNetCV(
    l1_ratio=[.1, .3, .5, .7, .9, .95, .99, 1.0],
    cv=5,
    random_state=42
)
elastic_net_cv.fit(X_train_scaled, y_train)

print(f"最优正则化强度(Alpha/Lambda): {elastic_net_cv.alpha_}")
print(f"最优 L1 比例 (l1_ratio): {elastic_net_cv.l1_ratio_}")

# 预测评估
y_pred = elastic_net_cv.predict(X_test_scaled)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"测试集均方误差 (MSE): {mse:.4f}")
print(f"决定系数 (R²): {r2:.4f}")

# 输出非零系数特征
coefficients = pd.Series(elastic_net_cv.coef_, index=X.columns)
print("\n非零系数特征（被模型保留的重要特征）：")
print(coefficients[coefficients != 0])
