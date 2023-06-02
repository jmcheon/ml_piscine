import numpy as np

def predict_(x, theta):
	"""
	Computes the vector of prediction y_hat from two non-empty numpy.array.

	Args:
	x: has to be an numpy.array, a vector of dimension m * 1.
	theta: has to be an numpy.array, a vector of dimension 2 * 1.

	Returns:
	y_hat as a numpy.array, a vector of dimension m * 1.
	None if x and/or theta are not numpy.array.
	None if x or theta are empty numpy.array.
	None if x or theta dimensions are not appropriate.

	Raises:
	This function should not raise any Exceptions.
	"""
	if not isinstance(x, np.ndarray) or not isinstance(theta, np.ndarray):
		print("Invalid input: arguments of ndarray type required")	
		return None

	if x.ndim == 1:
		x = x.reshape(x.size, 1)
	elif not (x.ndim == 2 and x.shape[1] == 1):
		print("Invalid input: wrong shape of x", x.shape)
		return None

	if theta.ndim == 1 and theta.size == 2:
		pass
	elif not (theta.ndim == 2 and theta.shape == (2, 1)):
		print("Invalid input: wrong shape of theta ", theta.shape)
		return None

	X = np.hstack((np.ones((x.shape[0], 1)), x))
	y_hat = X.dot(theta)
	return np.array(y_hat)

def ex1():
	x = np.arange(1,6)
	print("x:", x, x.shape)

	# Example 1:
	theta1 = np.array([[5], [0]])
	print(predict_(x, theta1)) # Ouput: array([[5.], [5.], [5.], [5.], [5.]])

	# Do you remember why y_hat contains only 5’s here?
	# Example 2:
	theta2 = np.array([[0], [1]])
	print(predict_(x, theta2)) # Output: array([[1.], [2.], [3.], [4.], [5.]])

	# Do you remember why y_hat == x here?
	# Example 3:
	theta3 = np.array([[5], [3]])
	print(predict_(x, theta3)) # Output: array([[ 8.], [11.], [14.], [17.], [20.]])

	# Example 4:
	theta4 = np.array([[-3], [1]])
	print(predict_(x, theta4)) # Output: array([[-2.], [-1.], [ 0.], [ 1.], [ 2.]])

def ex2():
	x = np.arange(1,6)
	print("x:", x, x.shape)

	# Example 1:
	theta1 = np.array([[5], [0], 2])
	print(predict_(x, theta1))

	# Example 2:
	theta2 = np.array([[0], [1], [2]])
	print(predict_(x, theta2))

	# Example 3:
	theta3 = np.array([[5], [3]])
	print(predict_(x, theta3))

if __name__ == "__main__":
	ex1()
