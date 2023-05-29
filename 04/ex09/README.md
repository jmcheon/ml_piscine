# Machine Learning Module 04 - ex09

<table>
<tr><th>Exercise 09 -  Practicing Regularized Logistic Regression</th></tr>
<tr><td>Turn-in directory : ex09/ </tr>
<tr><td>Files to turn in : solar_system_census.py, benchmark_train.py, models.[csv/yml/pickle]</tr>
<tr><td>Forbidden functions : sklearn</tr>
</table>

### Objective

It’s training time! Let’s practice our updated Logistic Regression with polynomial models.

### Introduction
You have already used the dataset **solar_system_census.csv** and **solar_system_census_planets.csv**

 - The dataset is divided in two files: **solar_system_census.csv** and **solar_system_census_planets.csv**.
 - The first file contains biometric information such as the height, weight, and bone density of several Solar System citizens
 - The second file contains the homeland of each citizen, indicated by its Space Zipcode representation (i.e. one number for each planet... :)).

As you should know, Solar citizens come from four registered areas (zipcodes):

 - The flying cities of Venus (0),
 - United Nations of Earth (1),
 - Mars Republic (2),
 - The Asteroids’ Belt colonies (3).

### Instructions
#### Split the Data
Take your **solar_system_census.csv** dataset and split it in a **training set**, a **crossvalidation set** and a **test set**.

#### Training and benchmark
One part of your submission will be find in **benchmark_train.py** and **models.[csv/yml/pickle]** files. You have to:

 - Train different regularized logistic regression models with a polynomial hypothesis of **degree 3**. The models will be trained with different λ values, ranging from 0 to 1. Use one-vs-all method.
 - Evaluate the **f1 score** of each of the models on the cross-validation set. You can use the f1_score_ function that you wrote in the **ex11** of **module03**.
 - Save the different models into a **models.[csv/yml/pickle]**.

#### Solar system census program
The second and last part of your submission is in **solar_system_census.py**. You have to:

 - Loads the differents models from **models.[csv/yml/pickle]** and train from scratch only the best one on a training set.
 - Visualize the performance of the different models with a bar plot showing the score of the models given their λ value.
 - Print the **f1 score** of all the models calculated on the test set.
 - Visualize the target values and the predicted values of the best model on the same scatterplot. Make some effort to have a readable figure.

For the second script solar_system_census.py, only a train and test set are necessary as one is simply looking to the performance.
