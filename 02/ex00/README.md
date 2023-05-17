# Machine Learning Module 02 - ex00

<table>
<tr><th>Exercise 00 -  Multivariate Hypothesis - Iterative Version</th></tr>
<tr><td>Turn-in directory : ex00/ </tr>
<tr><td>Files to turn in : prediction.py </tr>
<tr><td>Forbidden functions : None</tr>
</table>

### Objective

Manipulate the hypothesis to make prediction. You must implement the following formula as a function:
$$\hat{y}^{(i)} = \theta_0 + \theta_1 x_1^{(i)} + \ldots + \theta_n x_n^{(i)}   \quad \text{for } i = 1, \ldots, m$$

Where:

 - $\hat y$ is a vector of dimension $m$: the vector of predicted values,
 - $\hat y^{(i)}$ is the $i^{th}$ component of the $\hat y$ vector: the predicted value for the $i^{th}$ example,
 - $\theta$ is a vector of dimansion ($n$ + 1): the parameter vector,
 - $\theta_j$ is the $j^{th}$ component of the parameter vector,
 - $X$ is a matrix of dimensions ($m$ x  $n$): the design matrix,
 - $x^{(i)}$ is the $i^{th}$ row of the $X$ matrix: the featrue vector of the $i^{th}$ example,
 - $x_j$ is the $j^{th}$ column of the $X$ matrix,
 - $x_j^{(i)}$ is the element at the intersection of the $i^{th}$ row and the $j^{th}$ column of the $X$ matrix: the $j^{th}$ feature of the $i^{th}$ example
