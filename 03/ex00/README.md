# Machine Learning Module 03 - ex00

<table>
<tr><th>Exercise 00 -  Sigmoid</th></tr>
<tr><td>Turn-in directory : ex00/ </tr>
<tr><td>Files to turn in : sigmoid.py</tr>
<tr><td>Forbidden functions : None</tr>
</table>

### Objective

Introduction to the hypothesis notion in case of logistic regression. You must implement the sigmoid function, given by the following formula:
$$\text {sigmoid}(x) = \frac {1}{1 + e^{-x}}$$

Where:

 - $x$ is a scalar or a vector,
 - $e$ is the contracted form for exponential function. It is also a mathematical constant, named Euler’s number.

&emsp;This function is also known as **Standard logistic sigmoid function**. This explains the name *logistic regression*.
&emsp;The sigmoid function transforms an input into a probability value, i.e. a value between 0 and 1. This probability value will then be used to classify the inputs.
