import numpy as np
import os, sys

path = os.path.join(os.path.dirname(__file__), '..', 'ex03')
sys.path.insert(1, path)
from gradient import gradient

def fit_(x, y, theta, alpha, max_iter):
	"""
	Description:
	Fits the model to the training dataset contained in x and y.

	Args:
	x: has to be a numpy.array, a matrix of dimension m * n:
	(number of training examples, number of features).
	y: has to be a numpy.array, a vector of dimension m * 1:
	(number of training examples, 1).
	theta: has to be a numpy.array, a vector of dimension (n + 1) * 1:
	(number of features + 1, 1).
	alpha: has to be a float, the learning rate
	max_iter: has to be an int, the number of iterations done during the gradient descent

	Return:
	new_theta: numpy.array, a vector of dimension (number of features + 1, 1).
	None if there is a matching dimension problem.
	None if x, y, theta, alpha or max_iter is not of expected type.

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

	if not x.ndim == 2:
		print(f"Invalid input: wrong shape of x", x.shape)
		return None

	if y.ndim == 1:
		y = y.reshape(y.size, 1)
	elif not (y.ndim == 2 and y.shape[1] == 1):
		print(f"Invalid input: wrong shape of y", y.shape)
		return None
	
	if x.shape[0] != y.shape[0]:
		print(f"Invalid input: x, y matrices should be compatible.")
		return None

	if theta.ndim == 1 and theta.size == x.shape[1] + 1:
		theta = theta.reshape(x.shape[1] + 1, 1)
	elif not (theta.ndim == 2 and theta.shape == (x.shape[1] + 1, 1)):
		print(f"Invalid input: wrong shape of {theta}", theta.shape)
		return None

	# Weights to update: alpha * mean((y_hat - y) * x) 
	# Bias to update: alpha * mean(y_hat - y)

	new_theta = np.copy(theta.astype("float64"))
	for _ in range(max_iter):
		# Compute gradient descent
		grad = gradient(x, y ,new_theta)
		# Update new_theta
		for i in range(theta.size):
			new_theta[i] -= alpha * grad[i]

	return new_theta
	
def ex1():
	x = np.array([[0.2, 2., 20.], [0.4, 4., 40.], [0.6, 6., 60.], [0.8, 8., 80.]])
	y = np.array([[19.6], [-2.8], [-25.2], [-47.6]])
	theta = np.array([[42.], [1.], [1.], [1.]])
	
	# Example 0:
	theta2 = fit_(x, y, theta, alpha = 0.0005, max_iter=42000)
	print(theta2) # Output: array([[41.99..],[0.97..], [0.77..], [-1.20..]])
	
	# Example 1:
	X = np.hstack((np.ones((x.shape[0], 1)), x))
	y_hat = X.dot(theta2)
	print("\n", y_hat)
	#print(predict_(x, theta2)) # Output: array([[19.5992..], [-2.8003..], [-25.1999..], [-47.5996..]])

def ex2():
	X = np.array([[1., 1., 2., 3.], [5., 8., 13., 21.], [34., 55., 89., 144.]])
	Y = np.array([[23.], [48.], [218.]])
	theta = np.array([[1.], [1.], [1.], [1.], [1]])

	theta2 = fit_(X, Y, theta, alpha = 1.6e-4, max_iter=200000)
	print(theta2) # Output: array([[18.188..], [2.767..], [-0.374..], [1.392..], [0.017..]])

if __name__ == "__main__":
	ex1()
