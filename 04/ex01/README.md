# Machine Learning Module 04 - ex01

<table>
<tr><th>Exercise 01 - L2 Regularization</th></tr>
<tr><td>Turn-in directory : ex01/ </tr>
<tr><td>Files to turn in : l2_reg.py</tr>
<tr><td>Forbidden functions : sklearn</tr>
</table>

### Objective

You must implement the following formulas as functions:

#### Iterative
$$L_2{(\theta)^2} = \sum_{j=1}^{n} \theta_j^2$$
Where:

 - $\theta$ is a vector of dimension ($n$ + 1).
#### Vectorized
$$L_2{(\theta)^2} = \theta^{'} \cdot \theta^{'}$$
Where:
 - $\theta^{'}$ is a vector of dimension ($n$ + 1), constructed using the following rules: $$\theta_0^{'} = 0$$ $$\theta_j^{'} = \theta_j \quad \text {for } j = 1, \dots, n$$
