# Machine Learning Module 03 - ex03

<table>
<tr><th>Exercise 03 -  Vectorized Logistic Loss Function</th></tr>
<tr><td>Turn-in directory : ex03/ </tr>
<tr><td>Files to turn in : vec_log_loss.py</tr>
<tr><td>Forbidden functions : any function that performs derivatives for you</tr>
</table>

### Objective

Understanding and manipulation loss function concept in case of logistic regression. You must implement the following formula as a function: $$J(\theta) = - \frac {1}{m}[y \cdot log(\hat y) + (\vec 1 - y) \cdot log(\vec 1 - \hat y^)]$$

Where:

 - $\hat y$ is a vector of dimension $m$, the vector of predicted values,
 - $y$ is a vector of dimension $m$, the vector of expected values,
  - $\vec 1$ is a vector of dimension $m$, a vector full of ones.

The purpose of epsilon (eps) is to avoid $log$(0) errors, it is a very small residual value we add to y.
