import numpy as np

class MyRidge(ParentClass):
	"""
	Description: My personnal ridge regression class to fit like a boss.
	• __init__, special method, similar to the one you wrote in MyLinearRegression(module06),
	• get_params_, which get the parameters of the estimator,
	• set_params_, which set the parameters of the estimator,
	• loss_, which return the loss between 2 vectors (numpy arrays),
	• loss_elem_, which return a vector corresponding to the squared diffrence between 2 vectors (numpy arrays),
	• predict_, which generates predictions using a linear model,
	• gradient_, which calculates the vectorized regularized gradient,
	• fit_, which fits Ridge regression model to a training dataset.
	"""
	def __init__(self, thetas, alpha=0.001, max_iter=1000, lambda_=0.5):
		self.alpha = alpha
		self.max_iter = max_iter
		self.thetas = thetas
		self.lambda_ = lambda_
