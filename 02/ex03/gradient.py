import numpy as np

def gradient(x, y, theta):
	"""
	Computes a gradient vector from three non-empty numpy.array, without any for-loop.
	The three arrays must have the compatible dimensions.

	Args:
	x: has to be an numpy.array, a matrix of dimension m * n.
	y: has to be an numpy.array, a vector of dimension m * 1.
	theta: has to be an numpy.array, a vector (n +1) * 1.

	Return:
	The gradient as a numpy.array, a vector of dimensions n * 1,
	containg the result of the formula for all j.
	None if x, y, or theta are empty numpy.array.
	None if x, y and theta do not have compatible dimensions.
	None if x, y or theta is not of expected type.

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

	if theta.ndim == 1 and theta.size == x.shape[1] + 1:
		theta = theta.reshape(x.shape[1] + 1, 1)
	elif not (theta.ndim == 2 and theta.shape == (x.shape[1] + 1, 1)):
		print(f"Invalid input: wrong shape of {theta}", theta.shape)
		return None
		
	X = np.hstack((np.ones((x.shape[0], 1)), x))
	X_t = np.transpose(X)
	gradient = X_t.dot(X.dot(theta) - y) / len(y)
	return gradient 

def ex1():
	x = np.array([
	[ -6, -7, -9],
	[ 13, -2, 14],
	[ -7, 14, -1],
	[ -8, -4, 6],
	[ -5, -9, 6],
	[ 1, -5, 11],
	[ 9, -11, 8]])
	y = np.array([2, 14, -13, 5, 12, 4, -19]).reshape((-1, 1))

	theta1 = np.array([3,0.5,-6]).reshape((-1, 1))
	#print("x:", x.shape, "y:", y.shape, "t:", theta1.shape)
	
	# Example :
	#print(gradient(x, y, theta1)) # Output: array([[ -33.71428571], [ -37.35714286], [183.14285714], [-393.]])
	
	# Example :
	theta2 = np.array([0,0,0,0]).reshape((-1, 1))
	print(gradient(x, y, theta2)) # Output: array([[ -0.71428571], [ 0.85714286], [23.28571429], [-26.42857143]])

if __name__ == "__main__":
	ex1()
