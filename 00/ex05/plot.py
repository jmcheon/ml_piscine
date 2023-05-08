import numpy as np
import matplotlib.pyplot as plt
from tools import add_intercept

def plot(x, y, theta):
	"""
	Plot the data and prediction line from three non-empty numpy.array.

	Args:
	x: has to be an numpy.array, a vector of dimension m * 1.
	y: has to be an numpy.array, a vector of dimension m * 1.
	theta: has to be an numpy.array, a vector of dimension 2 * 1.

	Returns:
	Nothing.

	Raises:
	This function should not raise any Exceptions.
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

	fig, ax = plt.subplots()
	ax.scatter(x, y, color='blue', label='data points')
	ax.plot(x, y_hat, color='orange', label='prediction line')
	ax.legend()
	plt.show()

def ex1():
	x = np.arange(1,6)
	y = np.array([3.74013816, 3.61473236, 4.57655287, 4.66793434, 5.95585554])
	#x = x.reshape(x.size, 1)
	#y = y.reshape(y.size, 1)
	#print("x:", x, x.shape)
	#print("y:", y, y.shape)

	#X = np.hstack((x, y))
	#print("X:", X, X.shape)

	# Example 1:
	theta1 = np.array([[4.5],[-0.2]])
	plot(x, y, theta1)

	# Example 2:
	theta2 = np.array([[-1.5],[2]])
	plot(x, y, theta2)

	# Example 3:
	theta3 = np.array([[3],[0.3]])
	plot(x, y, theta3)

def ex2():
	plot(np.array([0, 1]), np.array([0, 1]), np.array([0, 1]))
	plot(np.array([0, 1]), np.array([0, 1]), np.array([1, 1]))
	plot(np.array([0, 2]), np.array([0, 0]), np.array([-1, 1]))

if __name__ == "__main__":
	ex1()
