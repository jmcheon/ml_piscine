# Machine Learning Module 01 - ex06

<table>
<tr><th>Exercise 06 -  Normalization II: Min-max Standardization</th></tr>
<tr><td>Turn-in directory : ex06/ </tr>
<tr><td>Files to turn in : minmax.py </tr>
<tr><td>Forbidden functions : None</tr>
</table>

### Objective

Introduction to standardization/normalization methods. Implement another normalization method.
You must implement the following formula as a function:
$$x^{'(i)} = \frac{x^{(i)} - min(x)} {max(x) - min(x)}  \quad \text{for } i = 1, \ldots, m$$

Where:

 - $x$ is a vector of dimension $m$,
 - $x^{(i)}$ is the $i^{th}$ component of the $x$ vector,
 - $min(x)$ is the minimum value found among the components of vector $x$,
 - $max(x)$ is the maximum value found among the components of vector $x$,

You will notice that this min-max standardization doesn’t scale the values to the [−1, 1] range. What do you think the final range will be?
