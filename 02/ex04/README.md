# Machine Learning Module 02 - ex04

<table>
<tr><th>Exercise 04 - Multivariate Gradient Descent</th></tr>
<tr><td>Turn-in directory : ex04/ </tr>
<tr><td>Files to turn in : fit.py </tr>
<tr><td>Forbidden functions : any function that calculates derivatives for you</tr>
</table>

### Objective

Understand and manipulate the concept of gradient descent in the case of multivariate linear regression. Implement a function to perform linear gradient descent (LGD) for multivariate linear regression.

### Instructions
In this exercise, you will implement linear gradient descent to fit your multivariate model to the dataset.

&emsp;The pseudocode of the algorithm is the following:

&emsp;repeat until convergence: { 
&emsp;&emsp;compute $∇(J)$ 
&emsp;&emsp;$θ := θ − α∇(J)$
&emsp;}


Where:

 - $∇(J)$ is the entire gradient vector,
 - $\theta$ is the entire parameter vector,
 - α (alpha) is the learning rate(a small number, usually between 0 and 1).

You are expected to write a function named fit_ as per the instructions
