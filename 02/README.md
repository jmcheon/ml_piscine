# Machine Learning Module 02 - Multivariate Linear Regression

> ***Summary :  Building on what you did on the previous modules you will extend the linear regression to handle more than one features. Then you will see how to build polynomial models and how to detect overfitting.***

| Exercise |                           Title                            |                         Description                          |
| :------: | :--------------------------------------------------------: | :----------------------------------------------------------: |
|    00    |        Multivariate Hypothesis - Iterative Version         |        Manipulate the hypothesis to make prediction. You must implement the following formula as a function: $$\hat{y}^{(i)} = \theta_0 + \theta_1 x_1^{(i)} + \ldots + \theta_n x_n^{(i)}   \quad \text{for } i = 1, \ldots, m$$         |
|    01    |        Multivariate hypothesis - vectorized version        |        Manipulate the hypothesis to make prediction.  |
|    02    |                  Vectorized Loss Function                  | Understand and manipulate loss function for multivariate linear regression. You must implement the following formula as a function: $$J(θ) = \frac{1}{2m} (\hat{y} - y)\cdot(\hat{y} - y) $$ |
|    03    |                Multivariate Linear Gradient                | Understand and manipulate concept of gradient in the case of multivariate formulation. You must implement the following formula as a function:$$\nabla (J) = \frac 1 m X^{'T}(X^{'}\theta - y)$$ |
|    04    |               Multivariate Gradient Descent                | Understand and manipulate the concept of gradient descent in the case of multivariate linear regression.<br /> Implement a function to perform linear gradient descent (LGD) for multivariate linear regression. |
|    05    |         Multivariate Linear Regression with Class          | Upgrade your Linear Regression class so it can handle multivariate hypotheses. |
|    06    |         Practicing Multivariate Linear Regression          | Fit a linear regression model to a dataset with multiple features. Plot the model’s predictions and interpret the graphs. |
|    07    |                     Polynomial models                      |    Broaden the comprehension of the notion of hypothesis. Create a function that takes a vector x of dimension m and an integer n as input, and returns a matrix of dimensions (m×n). Each column of the matrix contains x raised to the power of j, for j = 1, 2, ..., n: <br/>$x$ \| $x^2$ \| $x^3$ \| $\dots$ \| $x^n$ |
|    08    |               Let’s Train Polynomial Models!               | Manipulation of polynomial hypothesis. It’s training time! Let’s train some polynomial models, and see if those with higher polynomial degree perform better! |
|    09    |                        DataSpliter                         | Learn how to split a dataset into a training set and a test set. |
|    10    | Machine Learning for Grown-ups: Trantor guacamole business |            Let’s do Machine Learning for "real"!             |
