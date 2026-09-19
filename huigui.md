# 1. 代数证明题

设一元线性回归模型为

$$
y_i=\beta_0+\beta_1 x_i+e_i,\quad i=1,\dots,n
$$

其中残差为

$$
e_i=y_i-\hat{y}_i
$$

## （1）证明 $\displaystyle\sum_{i=1}^{n} e_i=0$

最小二乘法要最小化

$$
S(\beta_0,\beta_1)=\sum_{i=1}^{n}(y_i-\beta_0-\beta_1 x_i)^2
$$

对 $\beta_0$ 求偏导并令其为 0：

$$
\frac{\partial S}{\partial \beta_0}=-2\sum_{i=1}^{n}(y_i-\beta_0-\beta_1 x_i)=0
$$

于是

$$
\sum_{i=1}^{n}(y_i-\hat{\beta}_0-\hat{\beta}_1 x_i)=0
$$

即

$$
\sum_{i=1}^{n} e_i=0
$$

## （2）证明 $\displaystyle\sum_{i=1}^{n} x_i e_i=0$

对 $\beta_1$ 求偏导并令其为 0：

$$
\frac{\partial S}{\partial \beta_1}=-2\sum_{i=1}^{n}x_i(y_i-\beta_0-\beta_1 x_i)=0
$$

代入最优解得

$$
\sum_{i=1}^{n}x_i(y_i-\hat{\beta}_0-\hat{\beta}_1 x_i)=0
$$

即

$$
\sum_{i=1}^{n} x_i e_i=0
$$

所以在一元线性回归中，残差与常数项、自变量都正交。

# 2. 理解思考题

在多元回归中，若在模型中加入一个纯随机生成且与原变量无关的新特征，通常会出现：

- $R^2$ 上升或不变
- $R^2_{\text{adj}}$ 往往下降

## 原因解释

### 1）$R^2$ 为什么不会下降

$R^2$ 定义为

$$
R^2 = 1-\frac{RSS}{TSS}
$$

加入新变量后，模型的可选参数更多，最小化 $RSS$ 时的最优值不会变大，所以：

$$
RSS_{\text{new}} \le RSS_{\text{old}}
$$

因此

$$
R^2_{\text{new}} \ge R^2_{\text{old}}
$$

也就是说，$R^2$ 只会升高或保持不变。

### 2）$R^2_{\text{adj}}$ 为什么可能下降

调整后的决定系数为

$$
R^2_{\text{adj}} = 1-\frac{RSS/(n-p-1)}{TSS/(n-1)}
$$

其中 $p$ 是自变量个数。加入一个无关变量后：

- $RSS$ 可能只略微下降，甚至几乎不变；
- 但自由度 $n-p-1$ 变小了，相当于“惩罚”变强。

因此如果新增变量不能带来足够的解释能力提升，$R^2_{\text{adj}}$ 就会下降。

## 结论

- $R^2$：只看拟合优度，变量越多通常不会变差。
- $R^2_{\text{adj}}$：考虑模型复杂度，更适合比较不同变量数的模型。
