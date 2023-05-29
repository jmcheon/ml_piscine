# Machine Learning Module 04 - ex08

<table>
<tr><th>Exercise 08 - Regularized Logistic Regression</th></tr>
<tr><td>Turn-in directory : ex08/ </tr>
<tr><td>Files to turn in : my_logistic_regression.py</tr>
<tr><td>Forbidden functions : sklearn</tr>
</table>

### Objective

In the last exercise, you implemented of a regularized version of the linear regression algorithm, called Ridge regression. Now it’s time to update your logistic regression classifier as well! In the **scikit-learn** library, the logistic regression implementation offers a few regularization techniques, which can be selected using the parameter **penalty** (L2 is default). The goal of this exercise is to update your old **MyLogisticRegression** class to take that into account.

### Instructions

In the **my_logistic_regression.py** file, update your **MyLogisticRegression** class according to the following:

 - **add** a penalty parameter which can take the following values:’l2’, ’none’ (default value is ’l2’).
 - **update** the fit_(self, x, y) method:
	 - if penalty == ’l2’: use a **regularized version** of the gradient descent.
	 - if penalty = ’none’: use the **unregularized version** of the gradient descent from module03.
