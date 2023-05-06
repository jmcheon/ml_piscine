# Machine Learning Module 01 - ex00

<table>
<tr><th>Exercise 00 -  Linear Gradient - Iterative Version</th></tr>
<tr><td>Turn-in directory : ex00/ </tr>
<tr><td>Files to turn in : gradient.py </tr>
<tr><td>Forbidden functions : None</tr>
</table>

### Objective

Understand and manipulate the notion of gradient and gradient descent in machine learning. You must write a function that computes the gradient of the loss function. It must compute a partial derivative with respect to each theta parameter separately, and return the vector gradient. The partial derivatives can be calculated with the following formulas:
$$\nabla (J)_0 = \frac 1 m \sum_{i=1}^{m}(h_\theta (x^{(i)}) - y^{(i)})$$
$$\nabla (J)_1 = \frac 1 m \sum_{i=1}^{m}(h_\theta (x^{(i)}) - y^{(i)})x^{(i)}$$

Where:

 - $∇(J)$ is the gradient vector of size 2 × 1, (this strange symbol : $∇$ is called nabla)
 - $x$ is a vector of dimension $m$,
 - $y$ is a vector of dimension $m$,
 - $x^{(i)}$ is the $i^{th}$ component of vector $x$,
 - $y^{(i)}$ is the $i^{th}$ component of vector $y$,
 - $∇(J)_j$ is the $j^{th}$ component of vector $∇(J)$,
 - $h_θ(x ^{(i)} )$ corresponds to the model’s prediction of $y ^{(i)}$

### Hypothesis Notation

$h_θ(x ^{(i)} )$ is the same as what we previously noted $\hat y ^{(i)}$ . The two notations are equivalent. They represent the model’s prediction (or estimation) of the $y ^{(i)}$ value. If you follow Andrew Ng’s course material on Coursera, you will see him using the former notation.

&emsp; As a reminder: $h_θ(x^{(i)} ) = θ_0 + θ_1x^{(i)}$
