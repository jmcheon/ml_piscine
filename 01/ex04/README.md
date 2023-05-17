# Machine Learning Module 01 - ex04

<table>
<tr><th>Exercise 04 -  Practicing Linear Regression</th></tr>
<tr><td>Turn-in directory : ex04/ </tr>
<tr><td>Files to turn in : linear_model.py, are_blue_pills_magics.csv </tr>
<tr><td>Forbidden functions : sklearn</tr>
</table>

### Objective

Evaluate a linear regression model on a very small dataset, with a given hypothesis function $h$. Manipulate the loss function $J$, plot it, and briefly analyze the plot.

### Instructions
You can find in the **resources** folder a tiny dataset called **are_blue_pills_magics.csv** which gives you the driving performance of space pilots as a function of the quantity of the "blue pills" they took before the test. You have a description of the data in the file named **are_blue_pills_magics.txt**. As your hypothesis function $h$, you will choose:
$$h_\theta(x) = \theta_0 + \theta_1x$$

&emsp; Where $x$ is the variable, and $θ_0$ and $θ_1$ are the coefficients of the hypothesis. The hypothesis is a function of $x$. 

&emsp;**You are strongly encouraged to use the class you have implement in the previous exercise.**

Your program must:

 - Read the dataset from the csv file
 - perform a linear regression

Then you will model the data and plot 2 different graphs:

 - A graph with the data and the hypothesis you get for the spacecraft piloting score versus the quantity of "blue pills" 
 - The loss function $J(θ)$ in function of the $θ$ values
 - You will calculate the **MSE** of the hypothesis you chose (you know how to do it already).
