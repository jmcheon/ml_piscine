import numpy as np
import matplotlib.pyplot as plt
from tools import add_intercept
from loss import loss_

def plot_with_loss(x, y, theta):
	"""
	Plot the data and prediction line from three non-empty numpy.ndarray.

	Args:
	x: has to be an numpy.ndarray, a vector of dimension m * 1.
	y: has to be an numpy.ndarray, a vector of dimension m * 1.
	theta: has to be an numpy.ndarray, a vector of dimension 2 * 1.

	Returns:
	Nothing.

	Raises:
	This function should not raise any Exception.
	"""
	for v in [x, y, theta]:
		if not isinstance(v, np.ndarray):
			print(f"Invalid input: argument {v} of ndarray type required")	
			return

	if x.ndim == 1:
		x = x.reshape(x.size, 1)
	elif not (x.ndim == 2 and x.shape[1] == 1):
		print(f"Invalid input: wrong shape of {x}", x.shape)
		return

	if y.ndim == 1:
		y = y.reshape(y.size, 1)
	elif not (y.ndim == 2 and y.shape[1] == 1):
		print(f"Invalid input: wrong shape of {y}", y.shape)
		return
	
	if x.shape != y.shape:
		print("Invalid input: shapes of x, y should be indentical")
		return

	if theta.ndim == 1 and theta.size == 2:
		pass
	elif not (theta.ndim == 2 and theta.shape == (2, 1)):
		print("Invalid input: wrong shape of theta ", theta.shape)
		return

	X = add_intercept(x)
	y_hat = X.dot(theta)
	y_hat = y_hat.reshape(y_hat.size, 1)
	#cost = loss_(y, y_hat.reshape(y_hat.size, 1)) * 2
	cost = loss_(y, y_hat) * 2

	fig, ax = plt.subplots()
	ax.scatter(x, y, color='blue', label='data points')
	ax.plot(np.arange(1, y_hat.size + 1), y_hat, color='orange', label='prediction line')
	for xi, yi, y_hat_i in zip(x, y, y_hat):
		plt.plot([xi, xi], [yi, y_hat_i], '--', color='red')
	ax.legend()
	plt.title(f"Cost : {cost:.6f}")
	plt.show()


def ex1():
	x = np.arange(1,6)
	y = np.array([11.52434424, 10.62589482, 13.14755699, 18.60682298, 14.14329568])

	# Example 1:
	theta1= np.array([18,-1])
	plot_with_loss(x, y, theta1)
	
	# Example 2:
	theta2 = np.array([14, 0])
	plot_with_loss(x, y, theta2)
	
	# Example 3:
	theta3 = np.array([12, 0.8])
	plot_with_loss(x, y, theta3)

if __name__ == "__main__":
	ex1()
