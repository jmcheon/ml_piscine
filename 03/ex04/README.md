# Machine Learning Module 03 - ex04

<table>
<tr><th>Exercise 04 -  Logistic Gradient</th></tr>
<tr><td>Turn-in directory : ex04/ </tr>
<tr><td>Files to turn in : log_gradient.py</tr>
<tr><td>Forbidden functions : numpy and any function that performs derivatives for you</tr>
</table>

### Objective

Understand and manipulate concept of gradient in the case of logistic formulation. You must implement the following formula as a function: $$\nabla (J)_0 = \frac 1 m \sum(h_θ (x^{(i)}) - y^{(i)})$$
$$\nabla (J)_j = \frac 1 m \sum(h_θ (x^{(i)}) - y^{(i)})x_j^{(i)} \quad \text{for } j = 1, \ldots, n$$

Where:

 - $∇(J)$ is a vector of size ($n$ × 1),  the gradient vector.
 - $∇(J)_j$ is the $j^{th}$ component of vector $∇(J)$, the partial derivative of $J$ with respect to $\theta_j$.
 - $y$ is a vector of dimension $m$, the vector of expected values,
 - $y^{(i)}$ is a scalar, the $i^{th}$ component of vector $y$,
 - $x^{(i)}$ is the feature vector of $i^{th}$  example.
 - $x_j^{(i)}$ is a scalar, the $j^{th}$ feature value of the $i^{th}$ example.
 - $h_θ(x ^{(i)} )$ is a scalar, the model's extimation of $y^{(i)}$.

Remember that with logistic regression, the hypothesis is slightly different: $$h_θ(x^{(i)}) = sigmoid(\theta \cdot x^{'(i)})$$
