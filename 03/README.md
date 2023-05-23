# Machine Learning Module 03 - Logistic Regression

> ***Summary : Discover your first classification algorithm: logistic regression. You will learn its loss function, gradient descent and some metrics to evaluate its performance.***

| Exercise |               Title               |                         Description                          |
| :------: | :-------------------------------: | :----------------------------------------------------------: |
|    00    |              Sigmoid              | Introduction to the hypothesis notion in case of logistic regression. You must implement the sigmoid function, given by the following formula: $$\text {sigmoid}(x) = \frac {1}{1 + e^{-x}}$$ |
|    01    |        Logistic Hypothesis        | Introduction to the hypothesis notion in case of logistic regression. You must implement the following formula as a function: $$\hat y = \text {sigmoid}(X^{'}\cdot \theta) = \frac {1}{1 + e^{-X^{'}\cdot \theta}}$$ |
|    02    |      Logistic Loss Function       | Understanding and manipulation loss function concept in case of logistic regression. You must implement the following formula as a function: $$J(\theta) = - \frac {1}{m}[\sum_{i=1}^{m} y^{(i)}log(\hat y^{(i)}) + (1 - y^{(i)})log(1 - \hat y^{(i)})]$$ |
|    03    | Vectorized Logistic Loss Function | Understanding and manipulation loss function concept in case of logistic regression. You must implement the following formula as a function: $$J(\theta) = - \frac {1}{m}[y \cdot log(\hat y) + (\vec 1 - y) \cdot log(\vec 1 - \hat y^)]$$ |
|    04    |         Logistic Gradient         | Understand and manipulate concept of gradient in the case of logistic formulation. You must implement the following formula as a function: $$\nabla (J)_0 = \frac 1 m \sum(h_θ (x^{(i)}) - y^{(i)})$$  $$\nabla (J)_j = \frac 1 m \sum(h_θ (x^{(i)}) - y^{(i)})x_j^{(i)} \quad \text{for } j = 1, \ldots, n$$ |
|    05    |   Vectorized Logistic Gradient    | Understand and manipulate concept of gradient in the case of logistic formulation. You must implement the following formula as a function: $$\nabla (J) = \frac 1 m X^{'T}(h_θ (X) - y)$$ |
|    06    |        Logistic Regression        | Demonstrate your knowledge by implementing a logistic regression classifier using the gradient descent algorithm. <br />You must have seen the power of numpy for vectorized operations. |
|    07    |  Practicing Logistic Regression   | Now it’s time to test your Logistic Regression Classifier on real data!<br /> You will use the solar_system_census_dataset. |
|    08    |           Other metrics           | Understanding and manipulation of classification criteria (TP, FP, ...) and metrics.<br /> The goal of this exercise is to write four metric functions (which are also available in sklearn.metrics) and to understand what they measure and how they are constructed. |
|    09    |         Confusion Matrix          | Manipulation of confusion matrix concept.<br /> The goal of this exercise is to reimplement the function confusion_matrix available in<br /> sklearn.metrics and to learn what does the confusion matrix represent. |
