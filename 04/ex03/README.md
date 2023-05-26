# Machine Learning Module 04 - ex03

<table>
<tr><th>Exercise 03 - Regularized Logistic Loss Function</th></tr>
<tr><td>Turn-in directory : ex03/ </tr>
<tr><td>Files to turn in : logistic_loss_reg.py</tr>
<tr><td>Forbidden functions : sklearn</tr>
</table>

### Objective

You must implement the following formulas as functions: $$J{(\theta)} = -\frac{1}{m}[y \cdot log(\hat y) +(\vec1 - y) \cdot log(\vec 1 - \hat y)] + \frac{λ}{2m}(\theta^{'} \cdot \theta^{'})$$

Where:

 - $\hat y$ is a vector of dimension $m$, the predicted values,
 - $y$ is a vector of dimension $m$, the expected values,
 - $\vec 1$ is a vector of dimension $m$, a vector full of ones,
 - $λ$ is a constant, the regularization hyperparameter,
 - $\theta^{'}$ is a vector of dimension $n$, constructed using the following rules: $$\theta_0^{'} = 0$$ $$\theta_j^{'} = \theta_j \quad \text {for } j = 1, \dots, n$$
