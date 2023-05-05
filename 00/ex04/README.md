# Machine Learning Module 00 - ex04

<table>
<tr><th>Exercise 04 -  Prediction</th></tr>
<tr><td>Turn-in directory : ex04/ </tr>
<tr><td>Files to turn in : prediction.py </tr>
<tr><td>Forbidden functions : None</tr>
</table>

### Objective

Understand and manipulate the notion of hypothesis in machine learning. You must implement the following formula as a function: $$\hat{y}^{(i)} = \theta_0 + \theta_1 x^{(i)} \quad \text{for } i = 1, \ldots, m$$

Where:

 - $\hat{y}^{(i)}$ is the ${i}^{th}$ component of vector $\hat{y}$

 - $\hat{y}$ is a vector of dimension m, the vector of predicted values
 - $\theta$ is a vector of dimension 2 * 1, the vector of parameters
 -  ${x}^{(i)}$ is the ${i}^{th}$ component of vector ${x}$

 -   ${x}$ is a vector of dimension ${m}$, the vector of examples

But this time you have to do it with the linear algebra trick!
```math
$$\hat{y} = X^{'}\cdot\theta = \begin{bmatrix}
1 & {x}^{(1)} \\
\vdots & \vdots \\
1 & {x}^{(m)}\\
\end{bmatrix} \cdot \begin{bmatrix} 
\theta_{0} \\
\theta_{1} \\
\end{bmatrix} = \begin{bmatrix} 
\theta_{0} + \theta_1 x^{(1)}\\
\vdots \\
\theta_{0} + \theta_1 x^{(m)} \\
\end{bmatrix}$$
```

You have to transform x into $X^{'}$ to fit the dimension of θ!
