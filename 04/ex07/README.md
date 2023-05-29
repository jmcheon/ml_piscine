# Machine Learning Module 02 - ex10

<table>
<tr><th>Exercise 10 -  Machine Learning for Grown-ups: Trantor guacamole business</th></tr>
<tr><td>Turn-in directory : ex10/ </tr>
<tr><td>Files to turn in : space_avocado.py, benchmark_train.py, models.[csv/yml/pickle]</tr>
<tr><td>Forbidden functions : sklearn</tr>
</table>

### Objective

Let’s do Machine Learning for "real"!

### Introduction

The dataset is constituted of 5 columns:

 - **index**: not relevant,
 - **weight**: the avocado weight order (in ton),
 - **prod_distance**: distance from where the avocado ordered is produced (in Mkm),
 - **time_delivery**: time between the order and the receipt (in days),
 - **target**: price of the order (in trantorian unit).

It contains the data of all the avocado purchase made by Trantor administration (guacamole is a serious business there).

### Instructions
You have to explore different models and select the best you find. To do this:

 - Split your **space_avocado.csv** dataset into a training and a test set.
 - Use your **polynomial_features** method on your training set.
 - Consider several Linear Regression models with polynomial hypothesis with a maximum degree of 4.
 - Evaluate your models on the test set. 
According to your model evaluations, what is the best hypothesis you can get?
 - Plot the evaluation curve which help you to select the best model (evaluation metrics vs models).
 - Plot the true price and the predicted price obtain via your best model (3D representation or 3 scatterplots).

&emsp;The training of all your models can take a long time. Thus you need to train only the best one during the correction. But, you should return in **benchmark_train.py** the program which perform the training of all the models and save the parameters of the different models into a file. In **models.[csv/yml/pickle]** one must find the parameters of all the models you have explored and trained. In **space_avocado.py** train the model based on the best hypothesis you find and load the other models from **models.[csv/yml/pickle]**. Then evaluate and plot the different graphics as asked before.
