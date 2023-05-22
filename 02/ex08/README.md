# Machine Learning Module 02 - ex08

<table>
<tr><th>Exercise 08 -  Let’s Train Polynomial Models!</th></tr>
<tr><td>Turn-in directory : ex08/ </tr>
<tr><td>Files to turn in : polynomial_train.py</tr>
<tr><td>Forbidden functions : sklearn</tr>
</table>

### Objective

Manipulation of polynomial hypothesis. It’s training time! Let’s train some polynomial models, and see if those with higher polynomial degree perform better!
&emsp;Write a program which:

 - Reads and loads **are_blue_pills_magics.csv** dataset,
 - Trains six separate Linear Regression models with polynomial hypothesis with degrees ranging from 1 to 6,
 - Evaluates and prints evaluation score (MSE) of each of the six models,
 - Plots a bar plot showing the MSE score of the models in function of the polynomial degree of the hypothesis,
 - Plots the 6 models and the data points on the same figure. Use lineplot style for the models and scaterplot for the data points. Add more prediction points to have smooth curves for the models.

&emsp;You will use **Micrograms** as feature and **Score** as target. The implementation of the method **fit_** based on the simple gradient descent lakes of efficiency and sturdiness, which will lead to the impossibility of converging for polynomial models with high degree or with features having several orders of magnitude of difference. See the starting values for some thetas below to help you to get acceptable parameters values for the models. According to evaluation score only, what is the best hypothesis (or model) between the trained models? According to the last plot, why it is not true? Which phenomenom do you observed here?
