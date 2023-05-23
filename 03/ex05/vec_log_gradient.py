import numpy as np

def vec_log_gradient(x, y, theta):
	"""
	Computes a gradient vector from three non-empty numpy.ndarray, without any for-loop. The three arrays must have compatible
	
	Args:
	x: has to be an numpy.ndarray, a matrix of shape m * n.
	y: has to be an numpy.ndarray, a vector of shape m * 1.
	theta: has to be an numpy.ndarray, a vector (n +1) * 1.

	Returns:
	The gradient as a numpy.ndarray, a vector of shape n * 1, containg the result of the formula for all j.
	None if x, y, or theta are empty numpy.ndarray.
	None if x, y and theta do not have compatible shapes.

	Raises:
	This function should not raise any Exception.
	"""
	for v in [x, y, theta]:
		if not isinstance(v, np.ndarray):
			print(f"Invalid input: argument {v} of ndarray type required")	
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
		print(f"Invalid input: two vectors of compatible shape are required")
		return None

	if theta.ndim == 1 and theta.size == x.shape[1] + 1:
		theta = theta.reshape(x.shape[1] + 1, 1)
	elif not (theta.ndim == 2 and theta.shape == (x.shape[1] + 1, 1)):
		print(f"Invalid input: wrong shape of theta", theta.shape)
		return None

	X = np.hstack((np.ones((x.shape[0], 1)), x))
	y_hat = np.array(1 / (1 + np.exp(-X.dot(theta))))
	X_t = np.transpose(X)
	gradient = X_t.dot(y_hat - y) / len(y)
	return gradient

def ex1():	
	# Example 1:
	y1 = np.array([1]).reshape((-1, 1))
	x1 = np.array([4]).reshape((-1, 1))
	theta1 = np.array([[2], [0.5]])
	print(vec_log_gradient(x1, y1, theta1)) # Output: array([[-0.01798621], [-0.07194484]])
	# Example 2:
	y2 = np.array([[1], [0], [1], [0], [1]])
	x2 = np.array([[4], [7.16], [3.2], [9.37], [0.56]])
	theta2 = np.array([[2], [0.5]])
	print(vec_log_gradient(x2, y2, theta2)) # Output: array([[0.3715235 ], [3.25647547]])
	# Example 3:
	y3 = np.array([[0], [1], [1]])
	x3 = np.array([[0, 2, 3, 4], [2, 4, 5, 5], [1, 3, 2, 7]])
	theta3 = np.array([[-2.4], [-1.5], [0.3], [-1.4], [0.7]])
	print(vec_log_gradient(x3, y3, theta3)) # Output: array([[-0.55711039], [-0.90334809], [-2.01756886], [-2.10071291], [-3.27257351]])

if __name__ == "__main__":
	ex1()
