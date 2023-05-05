# Machine Learning Module 00 - ex02

<table>
<tr><th>Exercise 02 -  Simple Prediction</th></tr>
<tr><td>Turn-in directory : ex02/ </tr>
<tr><td>Files to turn in : prediction.py </tr>
<tr><td>Forbidden functions : any functions which performs prediction</tr>
</table>


### Objective

 Understand and manipulate the notion of hypothesis in machine learning. You must implement the following formula as a function: $$\hat{y}^{(i)} = \theta_0 + \theta_1 x^{(i)} \quad \text{for } i = 1, \ldots, m$$

Where:

 - ${x}$ is a vector of dimension ${m}$, the vector of examples/features (without the ${y}$ values)
 - $\hat{y}$ is a vector of dimension m * 1, the vector of predicted values
 - $\theta$ is a vector of dimension 2 * 1, the vector of parameters
 - ${y}^{(i)}$ is the ${i}^{th}$ component of vector ${y}$
 - ${x}^{(i)}$ is the ${i}^{th}$ component of vector ${x}$
