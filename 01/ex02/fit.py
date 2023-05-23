import os, sys
import numpy as np

path = os.path.join(os.path.dirname(__file__), '..', 'ex00')
sys.path.insert(1, path)
from gradient import simple_gradient

def fit_(x, y, theta, alpha, max_iter):
	"""
	Description:
	Fits the model to the training dataset contained in x and y.

	Args:
	x: has to be a numpy.ndarray, a vector of dimension m * 1: (number of training examples, 1).
	y: has to be a numpy.ndarray, a vector of dimension m * 1: (number of training examples, 1).
	theta: has to be a numpy.ndarray, a vector of dimension 2 * 1.
	alpha: has to be a float, the learning rate
	max_iter: has to be an int, the number of iterations done during the gradient descent

	Returns:
	new_theta: numpy.ndarray, a vector of dimension 2 * 1.
	None if there is a matching dimension problem.

	Raises:
	This function should not raise any Exception.
	"""
	# arguments type varification
	for v in [x, y, theta]:
		if not isinstance(v, np.ndarray):
			print(f"Invalid input: argument {v} of ndarray type required")	
			return None
	if not isinstance(alpha, float):
		print(f"Invalid input: argument alpha of float type required")	
		return None
	if not isinstance(max_iter, int):
		print(f"Invalid input: argument max_iter of int type required")	
		return None

	# vector arguments shape varification
	v = [x, y]
	for i in range(len(v)): 
		if v[i].ndim == 1:
			v[i] = v[i].reshape(v[i].size, 1)
		elif not (v[i].ndim == 2 and v[i].shape[1] == 1):
			print(f"Invalid input: wrong shape of {v[i]}", v[i].shape)
			return None
	x, y = v
	if theta.ndim == 1:
		theta = theta.reshape(theta.size, 1)
	elif not (theta.ndim == 2 and theta.shape == (2, 1)):
		print(f"Invalid input: wrong shape of {theta}", theta.shape)
		return None

	# Weights to update: alpha * mean((y_hat - y) * x) 
	# Bias to update: alpha * mean(y_hat - y)

	new_theta = np.copy(theta.astype("float64"))
	for _ in range(max_iter):
		# Compute gradient descent
		gradient = simple_gradient(x, y ,new_theta)

		# Update new_theta
		new_theta[0] -= alpha * gradient[0]
		new_theta[1] -= alpha * gradient[1]

	return new_theta

def ex1():
	x = np.array([[12.4956442], [21.5007972], [31.5527382], [48.9145838], [57.5088733]])
	y = np.array([[37.4013816], [36.1473236], [45.7655287], [46.6793434], [59.5585554]])
	#print("x:", x.shape, x.ndim)
	#print("y:", y.shape, y.ndim)
	theta= np.array([1, 1]).reshape((-1, 1))

	# Example 0:
	theta1 = fit_(x, y, theta, alpha=5e-8, max_iter=1500000)
	print(theta1) # Output: array([[1.40709365], [1.1150909 ]])

	# Example 1:
	X = np.hstack((np.ones((x.shape[0], 1)), x))
	y_hat = X.dot(theta1)
	print(np.array(y_hat)) # Output: array([[15.3408728 ], [25.38243697], [36.59126492], [55.95130097], [65.53471499]])
	#print(predict_(x, theta1)) # Output: array([[15.3408728 ], [25.38243697], [36.59126492], [55.95130097], [65.53471499]])
	
def ex2():
	x = np.array(range(1, 101)).reshape(-1, 1)
	y = 0.75 * x + 5
	theta = np.array([[1.],[1.]])
	print(fit_(x, y, theta, 1e-5, 20000))

if __name__ == "__main__":
	ex1()
