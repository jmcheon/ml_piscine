# Machine Learning Module 03 - ex08

<table>
<tr><th>Exercise 08 -  Other metrics</th></tr>
<tr><td>Turn-in directory : ex08/ </tr>
<tr><td>Files to turn in : other_metrics.py</tr>
<tr><td>Forbidden functions : None</tr>
</table>

### Objective

Understanding and manipulation of classification criteria (TP, FP, ...) and metrics. The goal of this exercise is to write four metric functions (which are also available in **sklearn.metrics**) and to understand what they measure and how they are constructed. 

You must implement the following fomulas: $$\text{accuracy} = \frac{tp + tn}{tp + fp + tn + fn}$$ $$\text{precision} = \frac{tp}{tp + fp}$$ $$\text{recall} = \frac{tp}{tp + fn}$$ $$\text{F1score} = \frac{2 \times \text{precision} \times \text{recall}}{\text{precision} + \text{recall}}$$

Where:

 - tp is the number of **true positives**,
 - fp is the number of **false positives**,
 - tn is the number of **true negatives**,
 - fn is the number of **false negatives**.

### Instructions

For the sake of simplicity, we will only ask you to use two parameters.
