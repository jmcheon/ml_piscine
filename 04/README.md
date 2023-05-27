
# Machine Learning Module 04 - Regularization

> _**Summary : Today you will fight overfitting! You will discover the concepts of regularization and how to implement it into the algorithms you already saw until now.**_

| Exercise |               Title               |                         Description                          |
| :------: | :-------------------------------: | :----------------------------------------------------------: |
|    00    |              Polynomial models II              | Create a function that takes a matrix $X$ of dimensions ($m$×$n$) and an integer $p$ as input, and returns a matrix of dimension ($m$ × ($np$)). For each column $x_j$ of the matrix $X$, the new matrix contains $x_j$ raised to the power of $k$, for $k$ = 1, 2, ..., $p$ : <br/> &ensp;&ensp;&ensp;&ensp; $x_1$ \| $\dots$ \| $x_n$ \| $x_1^2$ \| $\dots$ \| $x_n^2$ \| $\dots$  \| $x_1^p$ \| $\dots$ \| $x_n^p$ |
|    01    |        L2 Regularization        | You must implement the following formulas as functions: $$L_2{(\theta)^2} = \sum_{j=1}^{n} \theta_j^2$$ $$L_2{(\theta)^2} = \theta^{'} \cdot \theta^{'}$$|
|    02    |      Regularized Linear Loss Function       | You must implement the following formulas as functions: $$J{(\theta)} = \frac{1}{2m}[(\hat y - y) \cdot (\hat y - y) +  λ(\theta^{'} \cdot \theta^{'})]$$ |
|    03    | Regularized Logistic Loss Function | You must implement the following formulas as functions: $$J{(\theta)} = -\frac{1}{m}[y \cdot log(\hat y) +(\vec1 - y) \cdot log(\vec 1 - \hat y)] + \frac{λ}{2m}(\theta^{'} \cdot \theta^{'})$$ |
|    04    |         Regularized Linear Gradient         | You must implement the following formulas as a functions for the **linear regression hypothesis**: $$\nabla (J)_0 = \frac{1}{m}\sum(h_θ(x^{(i)}) - y^{(i)})$$ $$\nabla (J)_j = \frac{1}{m}(\sum(h_θ(x^{(i)}) - y^{(i)})x_j^{(i)} + \lambda θ_j) \quad \text {for } j = 1, \dots, n$$ $$\nabla (J) = \frac{1}{m}[X^{'T}(h_θ(X) - y) + \lambda θ^{'}]$$ |
|    05    |   Regularized Logistic Gradient    | You must implement the following formulas as a functions for the **logistic regression hypothesis**: $$\nabla (J)_0 = \frac{1}{m}\sum(h_θ(x^{(i)}) - y^{(i)})$$ $$\nabla (J)_j = \frac{1}{m}(\sum(h_θ(x^{(i)}) - y^{(i)})x_j^{(i)} + \lambda θ_j) \quad \text {for } j = 1, \dots, n$$ $$\nabla (J) = \frac{1}{m}[X^{'T}(h_θ(X) - y) + \lambda θ^{'}]$$ |
|    06    |        Ridge Regression        | Now it’s time to implement your **MyRidge** class, similar to the class of the same name in **sklearn.linear_model**. |
|    07    |  Practicing Ridge Regression   | It’s training time! Let’s practice our brand new Ridge Regression with a polynomial model.|
|    08    |          Regularized Logistic Regression            | In the last exercise, you implemented of a regularized version of the linear regression algorithm, called Ridge regression. Now it’s time to update your logistic regression classifier as well! In the scikit-learn library, the logistic regression implementation offers a few regularization techniques, which can be selected using the parameter penalty (L2 is default). The goal of this exercise is to update your old MyLogisticRegression class to take that into account. |
|    09    |     Practicing Regularized Logistic Regression              | It’s training time! Let’s practice our updated Logistic Regression with polynomial models. |
