# Machine Learning Module 00 - ex06

<table>
<tr><th>Exercise 06 -  Loss function</th></tr>
<tr><td>Turn-in directory : ex06/ </tr>
<tr><td>Files to turn in : loss.py </tr>
<tr><td>Forbidden functions : None</tr>
</table>


### Objective

Understand and manipulate the notion of loss function in machine learning. You must implement the following formula as a function (and another one very close to it):
$$J(θ) = \frac{1}{2m} \sum_{i=1}^{m} (\hat{y}^{(i)} - y^{(i)})^2 $$

Where:

 - $\hat{y}$ is a vector of dimension $m$, the vector of predicted values
 - $y$ is a vector of dimension $m$ * 1, the vector of expected values
 - $\hat{y}^{(i)}$ is the ${i}^{th}$ component of vector $\hat{y}$
 - ${y}^{(i)}$ is the ${i}^{th}$ component of vector ${y}$

### Instructions

The implementation of the loss function has been split in two functions:

 - loss_elem_(), which computes the squared distances for all examples (yˆ (i)−y (i) ) 2 ), 
 - loss_(), which averages the squared distances of all examples (the J(θ) above).


This loss function is very close to the one called "**Mean Squared Error**", which is frequently mentioned in Machine Learning resources. The difference is in the denominator as you can see in the formula of the $$MSE = \frac 1 m\sum_{i=1}^{m}(\hat y^{(i)} − y^{(i)})^2$$

Except the division by 2m instead of m, these functions are rigourously identical: $$J(θ) = \frac {MSE} 2$$ 

MSE is called like that because it represents the mean of the errors (i.e.: the differences between the predicted values and the true values), squared. 

You might wonder why we choose to divide by two instead of simply using the MSE? (It’s a good question, by the way.) 

 - First, it does not change the overall model evaluation: if all performance measures are divided by two, we can still compare different models and their performance ranking will remain the same. 
 - Second, it will be convenient when we will calculate the gradient tomorow. Be patient, and trust us ;)
