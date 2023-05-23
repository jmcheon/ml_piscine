# Machine Learning Module 03 - ex01

<table>
<tr><th>Exercise 01 -  Logistic Hypothesis</th></tr>
<tr><td>Turn-in directory : ex01/ </tr>
<tr><td>Files to turn in : log_pred.py</tr>
<tr><td>Forbidden functions : None</tr>
</table>

### Objective

Introduction to the hypothesis notion in case of logistic regression. You must implement the following formula as a function:
$$\hat y = \text {sigmoid}(X^{'}\cdot \theta) = \frac {1}{1 + e^{-X^{'}\cdot \theta}}$$

Where:

 - $X$ is a matrix of dimensions ($m$ x $n$), the design matrix,
 - $X^{'}$ is a matrix of dimensions ($m$ x ($n$ + 1)), the design matrix onto which a column of 1's is added as a first column,
 - $\hat y$ is a vector of dimension $m$, the vector of predicted values,
 - $\theta$ is a vector of dimension ($n$ + 1), the vector of parameters.
Be careful:
 - the $x$ your function will get as an input corresponds to $X$, the ($m$ x $n$) matrix. Not $X^{'}$.
 - $\theta$ is an ($n$ + 1) vector.
