在正交设计下，特征矩阵满足

$$
X^T X = I
$$

（其中 $I$ 为单位矩阵）。

OLS（普通最小二乘法）的闭合解公式为：

$$
\hat{\beta}_{OLS} = (X^T X)^{-1} X^T Y
$$

当 $X^T X = I$ 时，上式简化为：

$$
\hat{\beta}_{OLS} = I^{-1} X^T Y = X^T Y
$$

岭回归（Ridge）的闭合解公式为：

$$
\hat{\beta}_{Ridge} = (X^T X+\lambda I)^{-1} X^T Y
$$

将 $X^T X = I$ 代入上式：

$$
\begin{align*}
\hat{\beta}_{Ridge} &= (I+\lambda I)^{-1} X^T Y \\
&= \big((1+\lambda)I\big)^{-1} X^T Y
\end{align*}
$$

根据矩阵数乘求逆的性质

$$
\big((1+\lambda)I\big)^{-1} = \frac{1}{1+\lambda}I
$$

可得：

$$
\hat{\beta}_{Ridge} = \frac{1}{1+\lambda}I \cdot X^T Y
$$

由于前面已得出 $X^T Y=\hat{\beta}_{OLS}$，因此：

$$
\hat{\beta}_{Ridge} = \frac{1}{1+\lambda}\hat{\beta}_{OLS}
$$
