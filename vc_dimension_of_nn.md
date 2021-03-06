
# Advanced Machine Learning Homework 4

`注: 文章数学公式需要latex的支持`

出于证明的方便，本文将神经网络模型$o=w_2^T\sigma(W_1v+b_1)+b_2$视为成线性方程组，将问题转化为以矩阵的秩来探讨线性方程组解的结构。
对于有m组输入样例的神经网络，此问题可建立数学模型如下：

对于任意m维向量$y$，是否都存在m * d的矩阵$X$满足：

 (1) $\sigma(XW_1^T+B_1)w_2+b_2=Y$

其中$B_1$由m个相同向量$b_1$组成,  $b_2$为m * 1

由原神经网络的实际意义可知，m>d，且m>n

由线性方程组的定义可得，对于任意Y，当R(A)大于Y的维度时，$AX=Y$才有有非零解，我们以$R_{max}$来表示$R_{max}(AX)$来表示AX最大可能的秩

对比VC维度的定义，我们可以得到如下等式

$R_{max}(AX)$ = $D_{vc}(AX)$

因此原问题就转化成了求解矩阵最大可能的秩，也就是$R_{max}$

## Problem 1

将$\sigma(x)=x$代入(1)得：
$(XW_1^T+B_1)w_2+b_2=Y$

由矩阵的秩的定义可知：

$R_{max}(X)=min(m,d)=d$

$R_{max}(B_1)=1$ （全部为相同向量，因此秩为1）

$R_{max}(b_2)=min(m,1)=1$

$R_{max}(w_2)=min(n,1)=1$

### 当 n > d 时

$R_{max}(W_1)=min(d,n)=d$

则：$R_{max}(XW_1) = max(R(X), R(W_1)) = max(d, d) = d$

那么有：$R_{max}(XW_1^T + B_1) = R(XW_1) + R(B_1) = d + 1$

令$C=XW_1^T+B_1$，可得$C$为m * n矩阵，因此$R(C) = min(min(m, n), d+1) = min(n, d + 1) = d + 1$

所以有：$R_{max}((XW_1^T + B_1)w_2 + b_2) = max(R(XW_1^T + B_1), R(w2)) + 1 = (d + 1) + 1 = d + 2$

根据线性方程组解的结构可知，$Y$的维度，也就是m大于d+2时，该线性方程组没有非零解，因此可得，该神经网络在n > d时的VC维度为d+2

### 当 n <= d 时

$R_{max}(W_1)=min(d,n)=n$

同理可得$R_{max}(XW_1^T + B_1) = n + 1$

然而$R(C) = min(min(m, n), n + 1) = min(n, n + 1) = n$

所以$R_{max}((XW_1^T + B_1)w_2 + b_2) = n + 1$

### 小结

我们得到结论，当使用线性激活函数时，该神经网络的VC维度为min(n, d + 1) + 1，最终的VC维度受到输入维度的直接影响

## Problem 2

我们需要讨论一下通过$\sigma(x)=max(0, x)$后，矩阵的秩可能的变化

### 当 n > d 时

$R_{max}(XW_1^T + B_1) = d + 1$

令$C=XW_1^T+B_1$，此时C的存在d + 1个线性无关向量，即n - (d + 1)个向量能使用其他向量线性表示。

由于使得将$C$送入$\sigma(x)=max(0, x)$得到的$C'$改变了原本为负数的元素，则显然存在一个$C_0$，使得$C_0'$通过改变负元素，保留了原线性无关向量的线性无关性，却改变了其他向量，破坏其线性相关性。

由此我们可知，在m和n无限大的情况下，Relu的VC维度为无穷的结论。`（此处应使用数学归纳法予以展开，篇幅关系略过)`

那么显然有$R_{max}(\sigma(C))$>= n$

又C为m * n 矩阵，有$R_{max}(\sigma(C)) <= min(n, m) = n$, 因此$R_{max}(\sigma(C)) = n$

$R_{max}((XW_1^T + B_1)w_2 + b_2) = n + 1$

### 当 n <= d 时

$R_{max}(XW_1^T+B_1) = n + 1$

很明显，当XW_1^T + B_1元素都为正时，$\sigma(x) = x$退化为线性激活，因此$R_{max}(\sigma(XW_1^T + B_1)) = n + 1$

又$\sigma(XW_1^T + B_1)$为n * m矩阵，因此$R_{max}(\sigma(XW_1^T + B_1)) = min(n, m) = n$

因此$R_{max}(\sigma(XW_1^T + B_1)) = min(n, n + 1) = n$

也就是说$R_{max}((XW_1^T + B_1)w_2 + b_2) = n + 1$

### 小结

我们得到结论，当使用Relu激活函数时，该神经网络的VC维度为n + 1，最终的VC维度不受输入维度d的影响

## 结论

使用线性激活函数，其神经网络VC维度为min(n, d + 1) + 1

使用Relu激活函数，其神经网络VC维度为n + 1

对比线性函数以及Relu，可以很明显的感受到relu在输入向量维度不足时，很好的弥补了模型的VC维度。

也就是说，Relu提高了神经网络模型的容错率，使得我们参数特性不足时，仍能一定程度上拟合出不错的结果。
