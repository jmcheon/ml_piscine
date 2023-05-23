# Machine Learning Module 03 - ex05

<table>
<tr><th>Exercise 05 -  Vectorized Logistic Gradient</th></tr>
<tr><td>Turn-in directory : ex05/ </tr>
<tr><td>Files to turn in : vec_log_gradient.py</tr>
<tr><td>Forbidden functions : any function that performs derivatives for you</tr>
</table>

### Objective

Understand and manipulate concept of gradient in the case of logistic formulation. You must implement the following formula as a function: $$\nabla (J) = \frac 1 m X^{'T}(h_\theta (X) - y)$$

Where:

 - $∇(J)$ is the gradient vector of size ($n$ + 1),  
 - $X^{'}$ is a matrix of dimension($m$ x ($n$ + 1)), the design matrix onto which a column of ones was added as the first column.
 - $X^{'T}$ means the matrix has been transposed.
 - $h_θ(X)$ is a vacter of dimension $m$, the vector of predicted values.
 - $y$ is a vector of dimension $m$, the vector of expected values.
