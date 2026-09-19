import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import variance_inflation_factor

df = sm.datasets.get_rdataset("Carseats", "ISLR").data

print("===== 数据集前5行 =====")
print(df.head())

# ========== 1. 构建多元线性回归模型 ==========
model = smf.ols(formula="Sales ~ Price + Advertising + Population + Age + Income", data=df)
result = model.fit()

print("\n===== 回归模型结果 =====")
print(result.summary())

# ========== 2. VIF多重共线性检验 ==========
X = df[["Price", "Advertising", "Population", "Age", "Income"]]
X = sm.add_constant(X)

vif_data = pd.DataFrame()
vif_data["变量"] = X.columns
vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

print("\n===== 方差膨胀因子VIF =====")
print(vif_data)
