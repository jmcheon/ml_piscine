# Machine Learning Module 02 - ex01

<table>
<tr><th>Exercise 01 -  Multivariate hypothesis - vectorized version</th></tr>
<tr><td>Turn-in directory : ex01/ </tr>
<tr><td>Files to turn in : prediction.py </tr>
<tr><td>Forbidden functions : None</tr>
</table>

### Objective

Manipulate the hypothesis to make prediction. You must implement the following formula as a function:
```math
$$\hat{y} = X^{'}\cdot\theta = \begin{bmatrix}
1 & {x}_1^{(1)} & \ldots & x_n^{(1)} \\
\vdots & \vdots & \ddots & \vdots\\
1 & {x}_1^{(m)} & \ldots & x_n^{(m)} \\
\end{bmatrix} \cdot \begin{bmatrix} 
\theta_{0} \\
\theta_{1} \\
\vdots \\
\theta_n \\
\end{bmatrix} = \begin{bmatrix} 
\theta_{0} + \theta_1 x_1^{(1)} + \ldots +  \theta_n x_n^{(1)} \\
\vdots \\
\theta_{0} + \theta_1 x_1^{(m)} + \ldots +  \theta_n x_n^{(m)} \\
\end{bmatrix}$$
```

Where:

 - $\hat{y}$ is a vector of dimension $m$, the vector of predicted values,
 - $X$ is a matrix of dimensions ($m$ x $n$): the design matrix,
 - $X^{'}$ is a matrix of dimensions ($m$ x ($n$ + 1)): the design matrix onto which a column of 1's was added as a first column,
 - $\theta$ is a vector of dimension ($n$ + 1), the parameter vector, 
 -  ${x}^{(i)}$ is the ${i}^{th}$ row of the ${X}$ matrix, 
 -  ${x_j}$ is the ${j}^{th}$ column of the ${X}$ matrix, 
 - $x_j^{(i)}$ is the intersection of the $i^{th}$ row and the $j^{th}$ column of the $X$ matrix: the $j^{th}$ feature of the $i^{th}$ training example.

Be careful:

 - The $x$ argument your function will receive as an input corresponds to $X$, the ($m$ x $n$) matrix. Not $X^{'}$.
 - theta is an ($n$ + 1) vector.
 - You have to transform $x$ to fit theta's dimensions
