# Machine Learning Module 04 - ex06

<table>
<tr><th>Exercise 06 - Ridge Regression</th></tr>
<tr><td>Turn-in directory : ex06/ </tr>
<tr><td>Files to turn in : ridge.py</tr>
<tr><td>Forbidden functions : sklearn</tr>
</table>

### Objective

Now it’s time to implement your **MyRidge** class, similar to the class of the same name in **sklearn.linear_model**.

### Instructions

In the **ridge.py** file, create the following class as per the instructions given below: 
&emsp;Your **MyRidge** class will have at least the following methods:

 - __init__, special method, similar to the one you wrote in MyLinearRegression (module06),
 - **get_params_**, which get the parameters of the estimator,
 - **set_params_**, which set the parameters of the estimator,
 - **loss_**, which return the loss between 2 vectors (numpy arrays),
 - **loss_elem_**, which return a vector corresponding to the squared diffrence between 2 vectors (numpy arrays),
 - **predict_**, which generates predictions using a linear model,
 - **gradient_**, which calculates the vectorized regularized gradient,
 - **fit_**, which fits Ridge regression model to a training dataset.

You should consider inheritance from MyLinearRegression.

&emsp;If **MyRidge** inheritates from **MyLinearRegression**, you may not need to reimplement **predict_** method. 

&emsp;The difference between **loss_elem_**, **loss_**, **gradient_** and **fit_** methods implementation **MyRidge**’s and **MyLinearRegression** (implemented in module 02) is the use of a regularization term.
