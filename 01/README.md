# Machine Learning Module 01 - Univariate Linear Regression

> ***Summary :  Today you will implement a method to improve your model’s performance: gradient descent. Then you will discover the notion of normalization.***

| Exercise |                   Title                   |                         Description                          |
| :------: | :---------------------------------------: | :----------------------------------------------------------: |
|    00    |    Linear Gradient - Iterative Version    | Understand and manipulate the notion of gradient and gradient descent in machine learning.<br />You must write a function that computes the gradient of the loss function. $$\nabla (J)_0 = \frac 1 m \sum_{i=1}^{m}(h_\theta (x^{(i)}) - y^{(i)})$$ $$\nabla (J)_1 = \frac 1 m \sum_{i=1}^{m}(h_\theta (x^{(i)}) - y^{(i)})x^{(i)}$$ |
|    01    |   Linear Gradient - Vectorized Version    | Understand and manipulate the notion of gradient and gradient descent in machine learning. $$\nabla (J) = \frac 1 m X^{'T}(X^{'}\theta - y)$$ |
|    02    |             Gradient Descent              | Understand and manipulate the notion of gradient and gradient descent in machine learning.<br />Be able to explain what it means to fit a Machine Learning model to a dataset. Implement a function that performs Linear Gradient Descent (LGD). |
|    03    |       Linear Regression with Class        | Write a class that contains all methods necessary to perform linear regression. |
|    04    |       Practicing Linear Regression        | Evaluate a linear regression model on a very small dataset, with a given hypothesis function h. <br />Manipulate the loss function $J$, plot it, and briefly analyze the plot. |
|    05    | Normalization I: Z-score Standardization  |    Introduction to standardization/normalization methods. You must implement the following formula as a function:$$x^{'(i)} = \frac{x^{(i)} - \frac{1}{m}\sum x^{(i)}} {\sqrt{\frac {1}{m-1}\sum (x^{(i)} - \frac{1}{m}\sum x^{(i)})^2}}  \quad \text{for } i = 1, \ldots, m$$    |
|    06    | Normalization II: Min-max Standardization |    Introduction to standardization/normalization methods. Implement another normalization method. <br/>You must implement the following formula as a function: $$x^{'(i)} = \frac{x^{(i)} - min(x)} {max(x) - min(x)}  \quad \text{for } i = 1, \ldots, m$$ |
