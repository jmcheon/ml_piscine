# Machine Learning Module 01 - ex01

<table>
<tr><th>Exercise 01 -  Linear Gradient - Vectorized Version</th></tr>
<tr><td>Turn-in directory : ex01/ </tr>
<tr><td>Files to turn in : vec_gradient.py </tr>
<tr><td>Forbidden functions : None</tr>
</table>

### Objective

Understand and manipulate the notion of gradient and gradient descent in machine learning. You must implement the following formula as a function:
$$\nabla (J) = \frac 1 m X^{'T}(X^{'}\theta - y)$$

Where:

 - $∇(J)$ is a vector of size 2 × 1
 - $X^{'}$ is a **matrix** of dimension ($m$ x 2),
 - $X^{'T}$ is the transpose of $X^{'}$ . Its dimensions are (2 × $m$),
 - $y$ is a vector of dimension $m$,
 - $\theta$ is a vector of dimension (2 x 1)
 
 Be careful:
 
 - the $x$ you will get as an input is an m vector,
 - $θ$ is a 2 × 1 vector. You have to transform $x$ to fit the dimension of $θ$!
