import numpy as np

def predict_(x, theta):
	"""
	Computes the prediction vector y_hat from two non-empty numpy.array.

	Args:
	x: has to be an numpy.array, a vector of dimensions m * n.
	theta: has to be an numpy.array, a vector of dimensions (n + 1) * 1.

	Return:
	y_hat as a numpy.array, a vector of dimensions m * 1.
	None if x or theta are empty numpy.array.
	None if x or theta dimensions are not appropriate.
	None if x or theta is not of expected type.

	Raises:
	This function should not raise any Exception.
	"""
	if not isinstance(x, np.ndarray) or not isinstance(theta, np.ndarray):
		print("Invalid input: arguments of ndarray type required")	
		return None

	if not x.ndim == 2:
		print("Invalid input: wrong shape of x", x.shape)
		return None

	if theta.ndim == 1 and theta.size == x.shape[1] + 1:
		theta = theta.reshape(x.shape[1] + 1, 1)
	elif not (theta.ndim == 2 and theta.shape == (x.shape[1] + 1, 1)):
		print("Invalid input: wrong shape of theta ", theta.shape)
		return None
	
	X = np.hstack((np.ones((x.shape[0], 1)), x))
	y_hat = X.dot(theta)
	return np.array(y_hat)
	
def ex1():
	x = np.arange(1,13).reshape((4,-1))
	print("x:", x, x.shape)

	# Example 1:
	theta1 = np.array([5, 0, 0, 0]).reshape((-1, 1))
	print("\nexample1:")
	print(predict_(x, theta1)) # Ouput: array([[5.], [5.], [5.], [5.]])
	# Do you understand why y_hat contains only 5’s here?
	
	# Example 2:
	theta2 = np.array([0, 1, 0, 0]).reshape((-1, 1))
	print("\nexample2:")
	print(predict_(x, theta2)) # Output: array([[ 1.], [ 4.], [ 7.], [10.]])
	# Do you understand why y_hat == x[:,0] here?
	
	# Example 3:
	theta3 = np.array([-1.5, 0.6, 2.3, 1.98]).reshape((-1, 1))
	print("\nexample3:")
	print(predict_(x, theta3)) # Output: array([[ 9.64], [24.28], [38.92], [53.56]])
	
	# Example 4:
	theta4 = np.array([-3, 1, 2, 3.5]).reshape((-1, 1))
	print("\nexample4:")
	print(predict_(x, theta4)) # Output: array([[12.5], [32. ], [51.5], [71. ]])

if __name__ == "__main__":
	ex1()
