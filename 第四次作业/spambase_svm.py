import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ========== 1.加载数据集 ==========
# 数据集下载地址：Kaggle spambase dataset
# 列：前57列为特征，最后1列为标签 1=垃圾邮件 0=正常邮件
columns = [f"feat_{i}" for i in range(57)] + ["label"]
df = pd.read_csv("spambase.data", names=columns)

print("====数据集基础信息====")
print(f"样本总数量：{df.shape[0]}，特征数：{df.shape[1]-1}")
print(f"垃圾邮件样本数：{df[df['label']==1].shape[0]}")
print(f"正常邮件样本数：{df[df['label']==0].shape[0]}")

X = df.drop("label", axis=1)
y = df["label"]

# ==========2.划分训练集、测试集 7:3 ==========
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# ==========3.特征标准化【SVM必须！课件重点：SVM对特征缩放高度敏感】 ==========
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ==========4.训练两种核函数SVM：线性核 / RBF高斯核 ==========
print("\n=======【1】线性核SVM  C=1 =======")
svm_linear = SVC(kernel="linear", C=1, random_state=42)
svm_linear.fit(X_train_scaled, y_train)
y_pred_linear = svm_linear.predict(X_test_scaled)

print("测试集准确率：{:.4f}".format(accuracy_score(y_test, y_pred_linear)))
print("分类报告：")
print(classification_report(y_test, y_pred_linear))
print("混淆矩阵：")
print(confusion_matrix(y_test, y_pred_linear))

print("\n=======【2】RBF高斯核SVM C=10, gamma=0.01 =======")
svm_rbf = SVC(kernel="rbf", C=10, gamma=0.01, random_state=42)
svm_rbf.fit(X_train_scaled, y_train)
y_pred_rbf = svm_rbf.predict(X_test_scaled)

print("测试集准确率：{:.4f}".format(accuracy_score(y_test, y_pred_rbf)))
print("分类报告：")
print(classification_report(y_test, y_pred_rbf))
print("混淆矩阵：")
print(confusion_matrix(y_test, y_pred_rbf))

# =========输出支持向量数量（课件SVM稀疏性）==========
print("\n====稀疏性：支持向量数量====")
print(f"线性核SVM支持向量数目：{svm_linear.n_support_}")
print(f"RBF核SVM支持向量数目：{svm_rbf.n_support_}")
