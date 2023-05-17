# Machine Learning Module 02 - ex03

<table>
<tr><th>Exercise 03 -  Multivariate Linear Gradient</th></tr>
<tr><td>Turn-in directory : ex03/ </tr>
<tr><td>Files to turn in : gradient.py </tr>
<tr><td>Forbidden functions : None</tr>
</table>

### Objective

Understand and manipulate concept of gradient in the case of multivariate formulation. You must implement the following formula as a function:
$$\nabla (J) = \frac 1 m X^{'T}(X^{'}\theta - y)$$

Where:

 - $∇(J)$ is a vector of dimension ($n$ + 1), the gradient vector,
 - $X$ is a matrix of dimensions ($m$ x $n$), the design matrix,
 - $X^{'}$ is a matrix of dimension ($m$ x ($n$ + 1)), the design matrix onto which a column of 1's was added as a first column,
 - $\theta$ is a vector of dimension ($n$ + 1), the parameter vector,
 - $y$ is a vector of dimension $m$, the vector of expected values.
 
 Be careful:
 
 - the $x$ you will get as an input is an m vector,
 - $θ$ is a 2 × 1 vector. You have to transform $x$ to fit the dimension of $θ$!
