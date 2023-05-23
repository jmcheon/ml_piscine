# Machine Learning Module 03 - ex02

<table>
<tr><th>Exercise 02 -  Logistic Loss Function</th></tr>
<tr><td>Turn-in directory : ex02/ </tr>
<tr><td>Files to turn in : log_loss.py</tr>
<tr><td>Forbidden functions : None</tr>
</table>

### Objective

Understanding and manipulation loss function concept in case of logistic regression. You must implement the following formula as a function: $$J(\theta) = - \frac {1}{m}[\sum_{i=1}^{m} y^{(i)}log(\hat y^{(i)}) + (1 - y^{(i)})log(1 - \hat y^{(i)})]$$

Where:

 - $\hat y$ is a vector of dimension $m$, the vector of predicted values,
 - $\hat y^{(i)}$ is the $i^{th}$ component of the $\hat y$ vector,
 - $y$ is a vector of dimension $m$, the vector of expected values,
  - $y^{(i)}$ is the $i^{th}$ component of the $y$ vector,

The logarithmic function isn’t defined in 0. This means that if $y^{(i)}$ = 0 you will get an error when you try to compute $log(y^{(i)})$. The purpose of the eps argument is to avoid $log$(0) errors. It is a very small residual value we add to $y$.
