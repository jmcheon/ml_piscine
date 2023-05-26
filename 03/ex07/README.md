# Machine Learning Module 03 - ex07

<table>
<tr><th>Exercise 07 -  Practicing Logistic Regression</th></tr>
<tr><td>Turn-in directory : ex07/ </tr>
<tr><td>Files to turn in : mono_log.py, multi_log.py</tr>
<tr><td>Forbidden functions : sklearn</tr>
</table>

### Objective

Now it’s time to test your Logistic Regression Classifier on real data! You will use the **solar_system_census_dataset.**

### Instructions
Some words about the dataset:
 - You will work with data from the last Solar System Census.
 - The dataset is divided in two files: **solar_system_census.csv** and **solar_system_census_planets.csv**.
 - The first file contains biometric information such as the height, weight, and bone density of several Solar System citizens
 -  The second file contains the homeland of each citizen, indicated by its Space Zipcode representation (i.e. one number for each planet... :)).

As you should know, Solar citizens come from four registered areas (zipcodes):

 - The flying cities of Venus (0),
 - United Nations of Earth (1),
 - Mars Republic (2),
 - The Asteroids’ Belt colonies (3).
&emsp;You are expected to produce 2 programs that will use Logistic Regression to predict from which planet each citizen comes from, based on the other variables found in the census dataset.
