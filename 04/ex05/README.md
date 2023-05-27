# Machine Learning Module 04 - ex05

<table>
<tr><th>Exercise 05 - Regularized Logistic Gradient</th></tr>
<tr><td>Turn-in directory : ex05/ </tr>
<tr><td>Files to turn in : reg_logistic_grad.py</tr>
<tr><td>Forbidden functions : sklearn</tr>
</table>

### Objective

You must implement the following formulas as a functions for the **logistic regression hypothesis**:

#### Iterative
$$\nabla (J)_0 = \frac{1}{m}\sum_{i=1}^{m}(h_\theta(x^{(i)}) - y^{(i)})$$
$$\nabla (J)_j = \frac{1}{m}(\sum_{i=1}^{m}(h_\theta(x^{(i)}) - y^{(i)})x_j^{(i)} + \lambda\theta_j) \quad \text {for } j = 1, \dots, n$$

Where:

 - $\nabla(J)_j$ is the $j^{th}$ component of $\nabla(J)$,
 - $\nabla(J)$ is a vector of dimension ($n$ + 1), the gradient vector,
 - $m$ is a constant, the number of training examples used,
 - $h_\theta(x^{(i)})$ is the model's prediction for the $i^{th}$ training example,
 - $x^{(i)}$ is the feature vector (of dimension $n$) of the $i^{th}$ training example, found in the $i^{th}$ row of the $X$ matrix,
 - $X$ is a matrix of dimensions ($m$ x $n$), the design matrix,
 - $y^{(i)}$ is the $i^{th}$ component of the $y$ vector,
 - $y$ is a vector of dimension $m$, the expected values,
 - $λ$ is a constant, the regularization hyperparameter,
 - $\theta_j$ is the $j^{th}$ parameter of the $\theta$ vector,
 - $\theta$ is a vector of dimension ($n$ + 1), the parameter vector.


#### Vectorized

$$\nabla (J) = \frac{1}{m}[X^{'T}(h_\theta(X) - y) + \lambda\theta^{'}]$$

Where:

 - $\nabla(J)$ is a vector of dimension ($n$ + 1), the gradient vector,
 - $m$ is a constant, the number of training examples used,
 - $X$ is a matrix of dimensions ($m$ x $n$), the design matrix,
 - $X^{'}$ is a matrix of dimensions ($m$ x ($n$ + 1)), the design matrix onto which a column of ones is added as a first column,
 - $X^{'T}$ is the transpose of the matrix, with dimension (($n$ + 1) x $m$),
 - $h_\theta(X)$ is a vector of dimension $m$, the vector of predicted values, 

 - $y$ is a vector of dimension $m$, the expected values,
 - $λ$ is a constant, the regularization hyperparameter,
 - $\theta$ is a vector of dimension ($n$ + 1), the parameter vector.
 - $\theta^{'}$ is a vector of dimension ($n$ + 1), constructed using the following rules: $$\theta_0^{'} = 0$$ $$\theta_j^{'} = \theta_j \quad \text {for } j = 1, \dots, n$$
