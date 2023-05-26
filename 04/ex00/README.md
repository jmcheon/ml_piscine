# Machine Learning Module 04 - ex00

<table>
<tr><th>Exercise 00 - Polynomial models II</th></tr>
<tr><td>Turn-in directory : ex00/ </tr>
<tr><td>Files to turn in : polynomial_model_extended.py</tr>
<tr><td>Forbidden functions : sklearn</tr>
</table>

### Objective

Create a function that takes a matrix $X$ of dimensions ($m$×$n$) and an integer $p$ as input, and returns a matrix of dimension ($m$ × ($np$)). For each column $x_j$ of the matrix $X$, the new matrix contains $x_j$ raised to the power of $k$, for $k$ = 1, 2, ..., $p$ : 
&ensp;&ensp;&ensp;&ensp; $x_1$ \| $\dots$ \| $x_n$ \| $x_1^2$ \| $\dots$ \| $x_n^2$ \| $\dots$  \| $x_1^p$ \| $\dots$ \| $x_n^p$
