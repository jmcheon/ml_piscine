# Machine Learning Module 03 - ex06

<table>
<tr><th>Exercise 06 -  Logistic Regression</th></tr>
<tr><td>Turn-in directory : ex06/ </tr>
<tr><td>Files to turn in : my_logistic_regression.py</tr>
<tr><td>Forbidden functions : sklearn</tr>
</table>

### Objective

The time to use everything you built so far has come! Demonstrate your knowledge by implementing a logistic regression classifier using the gradient descent algorithm. You must have seen the power of **numpy** for vectorized operations. Well let’s make something more concrete with that.
&emsp;You may have to take a look at Scikit-Learn’s implementation of logistic regression and noticed that the **sklearn.linear_model.LogisticRegression** class offers a lot of options.
&emsp;The goal of this exercise is to make a simplified but nonetheless useful and powerful version, with fewer options.

### Instructions
In the **my_logistic_regression.py** file, write a **MyLogisticRegression** class as in the instructions below:

You will add at least the following methods: 
 - predict_(self, x) 
 - loss_elem_(self, y, yhat)
 - loss_(self, y, yhat) 
 -  fit_(self, x, y)

You have already written these functions, you will just need few adjustments so that they all work well within your **MyLogisticRegression** class.
