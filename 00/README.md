
# Machine Learning Module 00 - Stepping into Machine Learning

> ***Summary : You will start by reviewing some linear algebra and statistics. Then you will implement your first model and learn how to evaluate its performances.***

| Exercise |           Title            |                         Description                          |
| :------: | :------------------------: | :----------------------------------------------------------: |
|    00    |         The Matrix         |  Manipulation and understanding of basic matrix operations.  |
|    01    |      TinyStatistician      |         Initiation to very basic statistic notions.          |
|    02    |     Simple Prediction      | Understand and manipulate the notion of hypothesis in machine learning.<br/>You must implement the following formula as a function:<br/> $$\hat{y}^{(i)} = \theta_0 + \theta_1 x^{(i)} \quad \text{for } i = 1, \ldots, m$$|
|    03    |       Add Intercept        | Understand and manipulate the notion of hypothesis in machine learning.<br/>You must implement a function which adds an extra column of 1’s on the left side of a given vector or matrix. |
|    04    |         Prediction         | Understand and manipulate the notion of hypothesis in machine learning.<br/>You must implement the following formula as a function:<br/>  $$\hat{y}^{(i)} = \theta_0 + \theta_1 x^{(i)} \quad \text{for } i = 1, \ldots, m$$|
|    05    |   Let’s Make Nice Plots    | You must implement a function to plot the data and the prediction line (or regression line).<br/>You will plot the data points (with their x and y values), and the prediction line that represents your hypothesis (h<sub>θ</sub>). |
|    06    |       Loss function        | Understand and manipulate the notion of loss function in machine learning.<br/>You must implement the following formula as a function (and another one very close to it):<br/> $$J(θ) = \frac{1}{2m} \sum_{i=1}^{m} (\hat{y}^{(i)} - y^{(i)})^2 $$ |
|    07    |  Vectorized loss function  | Understand and manipulate the notion of loss function in machine learning.<br/>You must implement the following formula as a function:<br/>$$J(θ) = \frac{1}{2m}(\hat{y} - y)\cdot(\hat{y} - y) $$ |
|    08    | Lets Make Nice Plots Again | You must implement a function which plots the data, the prediction line, and the loss. |
|    09    |    Other loss functions    |   Deepen the notion of loss function in machine learning.    |


# Exercise 02
## Interlude - Predict, Evaluate, Improve

### Predict

#### A very simple model
We have some data. We want to model it.

 - First we need to make an assumption, or hypothesis, about the structure of the data and the relationship between the variables.
 - Then we can apply that hypothesis to our data to make predictions.
  $$hypothesis(data) = predictions$$

#### Hypothesis
Let’s start with a very simple and intuitive **hypothesis** on how the price of a spaceship can be predicted based on the power of its engines. 
We will consider that the more powerful the engines are, the more expensive the spaceship is.
Furthermore, we will assume that the price increase is **proportional** to the power increase. In other words, we will look for a **linear relationship** between the two variables.
This means that we will formulate the price prediction with a linear equation, that you might be already familiar with:
$$\hat y = ax + b$$

We add the ˆ symbol over the y to specify that $\hat y$ (pronounced y-hat) is a **prediction** (or estimation) of the real value of $y$. The prediction is calculated with the **parameters** $a$ and $b$ and the input value $x$.
For example, if $a = 5$ and $b = 33,$ then $\hat y = 5x + 33.$
But in Machine Learning, we don’t like using the letters a and b. Instead we will use the following notation:
$$ \hat y = \theta_0 + \theta_1 x$$
So if $θ_0 = 33$ and $θ_1 = 5$, then $\hat y = 33 + 5x.$
To recap, this linear equation is our **hypothesis**. Then, all we will need to do is find the right values for our parameters $θ_0$ and $θ_1$ and we will get a fully-functional **prediction model**.


#### Predictions
Now, how can we generate a set of predictions on an entire dataset? Let’s consider a dataset containing $m$ data points (or space ships), called **examples**.
What we do is stack the $x$ and $\hat y$ values of all examples in vectors of length $m$. The relation between the elements in our vectors can then be represented with the following formula:
$$\hat{y}^{(i)} = \theta_0 + \theta_1 x^{(i)} \quad \text{for } i = 1, \ldots, m$$

Where:
 - $\hat y^{(i)}$ is the $i^{th}$ component of vector $\hat y$,
 - $x$ is the $i^{th}$ component of vector $x$,

Which can be experessed as:
```math
$$\hat y = \begin{bmatrix} 
\theta_0 + \theta_1 × x^{(1)} \\
\vdots \\
\theta_0 + \theta_1 × x^{(m)} \\
\end{bmatrix}$$
```

For example,
```math
 $$Given\quadθ = \begin{bmatrix}  
 33 \\ 5 \\ 
 \end{bmatrix}
 and  \quad x =  \begin{bmatrix}  
 1 \\ 3 \\ 
 \end{bmatrix}$$

 $$\hat y = h_\theta(x) = \begin{bmatrix}  
 33 + 5 × 1\\ 33 + 5 × 3 \\ 
 \end{bmatrix}
  =  \begin{bmatrix}  
 38 \\ 48 \\ 
 \end{bmatrix}$$
 ```

### More information

#### Why the $θ$ notation?
You might have two questions at the moment:

 - **WTF is that weird symbol?** This strange symbol, $θ$, is called "theta".
 - **Why use this notation instead of $a$ and $b$, like we’re used to?** Despite its seeming more complicated at first, the theta notation is actually meant to simplify your equations later on. Why? a and b are good for a model with two parameters, but you will soon need to build more complex models that take into account more variables than just x. You could add more letters like this: $\hat y = ax_1 + bx_2 + cx_3 + ... + yx_25 + z$ But how do you go beyond 26 parameters? And how easily can you tell what parameter is associated with, let’s say, $x_{19}$? That’s why it becomes more handy to describe all your parameters using the theta notation and indices. With $θ$, you just have to increment the number to name the parameter: $\hat y = θ_0 + θ_1x_1 + θ_2x_2 + ... + θ_{2468}x_{2468} ...$ Easy right?

#### Another common notation
$$\hat y = h_\theta(x)$$
Because $\hat y$ is calculated with our linear hypothesis using $θ$ and $x$, it is sometimes written as $h_θ(x)$. The $h$ stands for $hypothesis$, and can be read as " $the$ $result$ $of$ $our$ $hypothesis$ $h$ $given$ $x$ $and$ $theta$".
Then if x = 7, we can calculate: $\hat y = h_θ(x) = 33 + 5 × 7 = 68$ We can now say that according to our linear model, the **predicted value** of y given (x = 7) is 68.

# Exercise 03
## Interlude - A Simple Linear Algebra Trick
As you know, vectors and matrices can be multiplied to perform linear combinations. Let’s do a little linear algebra trick to optimize our calculation and use matrix multiplication. If we add a column full of 1’s to our vector of examples $x$, we can create the following matrix:
```math
$$X^{'} =  \begin{bmatrix}
1  & x^{(1)} \\ 
\vdots & \vdots \\
1 &x^{(m)} \\
\end{bmatrix}
$$
```
We can then rewrite our hypothesis as:
```math
$$\hat y^{(i)} = θ\cdot x^{'(i)} = 
\begin{bmatrix}
\theta_0\\ 
\theta_1\\ 
\end{bmatrix}
\cdot
\begin{bmatrix}
1 & x^{(i)}\\ 
\end{bmatrix} = \theta_0 + \theta_1 x^{(i)}
$$
```
Therefore, the calculation of each $\hat y^{(i)}$ can be done with only one vector multiplication. 
But we can even go further, by calculating the whole $\hat y$ vector in one operation:
```math
$$\hat y = X^{'} =  \begin{bmatrix}
1  & x^{(1)} \\ 
\vdots & \vdots \\
1 &x^{(m)} \\
\end{bmatrix}\cdot
\begin{bmatrix}
\theta_0\\ 
\theta_1\\ 
\end{bmatrix} = \begin{bmatrix} 
\theta_0 + \theta_1 x^{(1)} \\
\vdots \\
\theta_0 + \theta_1 x^{(m)} \\
\end{bmatrix}
$$
```
We can now get to the same result as in the previous exercise with just a single multiplication between our brand new $X^{'}$ matrix and the $θ$ vector!

#### A Note on Notation
In further Interludes, we will use the following convention:

 - Capital letters represent matrices (e.g.: $X$)
 - Lower case letters represent vectors and scalars (e.g.: $x^{(i)} , y$)

# Exercise 06
## Interlude - Evaluate

### Introducing the loss function
How good is our model? It is hard to say just by looking at the plot. We can clearly observe that certain regression lines seem to fit the data better than others, but it would be convenient to find a way to measure it.

To evaluate our model, we are going to use a **metric** called **loss function** (sometimes called **cost function**). The loss function tells us how bad our model is, how much it costs us to use it, how much information we lose when we use it. If the model is good, we won’t lose that much, if it’s terrible, we have a high loss!

The metric you choose will deeply impact the evaluation (and therefore also the training) of your model.

A frequent way to evaluate the performance of a regression model is to measure the distance between each predicted value ($\hat y^{(i)}$) and the real value it tries to predict ($y^{(i)}$). The distances are then squared, and averaged to get one single metric, denoted $J$:
$$J(θ) = \frac{1}{2m} \sum_{i=1}^{m} (\hat{y}^{(i)} - y^{(i)})^2 $$
The smaller, the better!

# Exercise 08
## Interlude - Fifty Shades of Linear Algebra

In the last exercise, we implemented the loss function in two subfunctions. It worked, but it’s not very pretty. What if we could do it all in one step, with linear algebra?

As we did with the hypothesis, we can use a vectorized equation to improve the calculations of the loss function.

So now let’s look at how squaring and averaging can be performed (more or less) in a single matrix multiplication!
$$J(θ) = \frac{1}{2m} \sum_{i=1}^{m} (\hat{y}^{(i)} - y^{(i)})^2 $$
$$J(θ) = \frac{1}{2m} \sum_{i=1}^{m} [(\hat{y}^{(i)} - y^{(i)})(\hat{y}^{(i)} - y^{(i)})] $$

Now, if we apply the definition of the dot product:
$$J(θ) = \frac{1}{2m} (\hat{y} - y)\cdot(\hat{y} - y) $$
