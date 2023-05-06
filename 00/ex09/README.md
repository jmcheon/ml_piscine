# Machine Learning Module 00 - ex09

<table>
<tr><th>Exercise 09 -  Other loss functions</th></tr>
<tr><td>Turn-in directory : ex09/ </tr>
<tr><td>Files to turn in : other_losses.py </tr>
<tr><td>Forbidden functions : None</tr>
</table>

Deepen the notion of loss function in machine learning. 

You certainly had a lot of fun implementing your loss function. Remember we told you it was **one among many possible ways of measuring the loss**. Now, you will get to implement other metrics. You already know about one of them: **MSE**. There are several more which are quite common: **RMSE**, **MAE** and **R2score**

### Objective

You must implement the following formulas as functions:
$$MSE(y,\hat y) = \frac 1 m \sum_{i=1}^{m}(\hat y^{(i)} - y^{(i)})^2$$
$$RMSE(y,\hat y) = \sqrt{\frac 1 m \sum_{i=1}^{m}(\hat y^{(i)} - y^{(i)})^2}$$
$$MAE(y,\hat y) = \frac 1 m \sum_{i=1}^{m}|\hat y^{(i)} - y^{(i)}|$$
$$R^2(y,\hat y) = 1 - \frac {\sum_{i=1}(\hat y^{(i)} - y^{(i)})^2}{\sum_{i=1}(y^{(i)} - \bar y)^2}$$

Where:

 - $y$ is a vector of dimension $m$,
 - $\hat y$ is a vector of dimension $m$,
 - $y^{(i)}$ is the $i^{th}$ component of vector $y$,
 - $\hat y^{(i)}$ is the $i^{th}$ component of vector $\hat y$,
 - $\bar y$ is the mean of the $y$ vector
