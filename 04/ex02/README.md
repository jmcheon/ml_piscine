# Machine Learning Module 04 - ex02

<table>
<tr><th>Exercise 02 - Regularized Linear Loss Function</th></tr>
<tr><td>Turn-in directory : ex02/ </tr>
<tr><td>Files to turn in : linear_loss_reg.py</tr>
<tr><td>Forbidden functions : sklearn</tr>
</table>

### Objective

You must implement the following formulas as functions: $$J{(\theta)} = \frac{1}{2m}[(\hat y - y) \cdot (\hat y - y) +  λ(\theta^{'} \cdot \theta^{'})]$$
Where:

 - $y$ is a vector of dimension $m$, the expected values,
 - $\hat y$ is a vector of dimension $m$, the predicted values,
 - $λ$ is a constant, the regularization hyperparameter,
 - $\theta^{'}$ is a vector of dimension $n$, constructed using the following rules: $$\theta_0^{'} = 0$$ $$\theta_j^{'} = \theta_j \quad \text {for } j = 1, \dots, n$$
