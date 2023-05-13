# Machine Learning Module 01 - ex02

<table>
<tr><th>Exercise 02 -  Gradient Descent</th></tr>
<tr><td>Turn-in directory : ex02/ </tr>
<tr><td>Files to turn in : fit.py </tr>
<tr><td>Forbidden functions : any function that calculates derivatives for you</tr>
</table>

### Objective

Understand and manipulate the notion of gradient and gradient descent in machine learning. Be able to explain what it means to **fit** a Machine Learning model to a dataset. Implement a function that performs **Linear Gradient Descent (LGD).**

Where:

 - $∇(J)$ is a vector of size 2 × 1
 - $X^{'}$ is a **matrix** of dimension ($m$ x 2),
 - $X^{'T}$ is the transpose of $X^{'}$ . Its dimensions are (2 × $m$),
 - $y$ is a vector of dimension $m$,
 - $\theta$ is a vector of dimension (2 x 1)
 
 Be careful:
 
 - the $x$ you will get as an input is an m vector,
 - $θ$ is a 2 × 1 vector. You have to transform $x$ to fit the dimension of $θ$!

### Instructions
In this exercise, you will implement linear gradient descent to fit your model to the dataset.

&emsp;The pseudocode for the algorithm is the following:


&emsp;repeat until convergence: { 
  
&emsp;&emsp;compute $∇(J)$ 
   
&emsp;&emsp;$θ_0 := θ_0 − α∇(J)_0$
   
&emsp;&emsp;$θ_1 := θ_1 − α∇(J)_1$
   
&emsp;}


Where:

 - α (alpha) is the learning rate. It’s a small float number (usually between 0 and 1),
 - For now, "reapeat until convergence" will mean to simply repeat for max_iter (a number that you will choose wisely).
