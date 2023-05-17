# Machine Learning Module 01 - ex05

<table>
<tr><th>Exercise 05 -  Normalization I: Z-score Standardization</th></tr>
<tr><td>Turn-in directory : ex05/ </tr>
<tr><td>Files to turn in : z_score.py </tr>
<tr><td>Forbidden functions : None</tr>
</table>

### Objective

Introduction to standardization/normalization methods. You must implement the following formula as a function:
$$x^{'(i)} = \frac{x^{(i)} - \frac{1}{m}\sum x^{(i)}} {\sqrt{\frac {1}{m-1}\sum (x^{(i)} - \frac{1}{m}\sum x^{(i)})^2}}  \quad \text{for } i = 1, \ldots, m$$

Where:

 - $x$ is a vector of dimension $m$,
 - $x^{(i)}$ is the $i^{th}$ component of the $x$ vector,
 - $x^{'}$ is the normalized version of the $x$ vector,

The equation is much easier to understand in the following form:
$$x^{'(i)} = \frac{x^{(i)} - \mu}{\sigma}  \quad \text{for } i = 1, \ldots, m$$

This should remind you something from **TinyStatistician**... 
Nope? 
Ok let’s do a quick recap:

 - $\mu$ is the mean of $x$,
 - $\sigma$ is the standard deviation of $x$.
