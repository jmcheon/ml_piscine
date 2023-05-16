import numpy as np

def gradient(x, y, theta):
	"""
	Computes a gradient vector from three non-empty numpy.array, without any for loop.
	The three arrays must have compatible shapes.

	Args:
	x: has to be a numpy.array, a matrix of shape m * 1.
	y: has to be a numpy.array, a vector of shape m * 1.
	theta: has to be a numpy.array, a 2 * 1 vector.

	Return:
	The gradient as a numpy.ndarray, a vector of dimension 2 * 1.
	None if x, y, or theta is an empty numpy.ndarray.
	None if x, y and theta do not have compatible dimensions.

	Raises:
	This function should not raise any Exception.
	"""
	for v in [x, y, theta]:
		if not isinstance(v, np.ndarray):
			print(f"Invalid input: argument {v} of ndarray type required")	
			return None

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
		
	# We add a column of 1's for the column of interception
	X = np.hstack((np.ones((x.shape[0], 1)), x))
	#print("X:", X, X.shape)
	X_t = np.transpose(X)
	#print("X_t:", X_t, X_t.shape)

	gradient = X_t.dot(X.dot(theta) - y) / len(y)
	return gradient 

def ex1():
	x = np.array([12.4956442, 21.5007972, 31.5527382, 48.9145838, 57.5088733]).reshape((-1, 1))
	y = np.array([37.4013816, 36.1473236, 45.7655287, 46.6793434, 59.5585554]).reshape((-1, 1))
	
	# Example 0:
	theta1 = np.array([2, 0.7]).reshape((-1, 1))
	print(gradient(x, y, theta1)) # Output: array([[-19.0342...], [-586.6687...]])
	
	# Example 1:
	theta2 = np.array([1, -0.4]).reshape((-1, 1))
	print(gradient(x, y, theta2)) # Output: array([[-57.8682...], [-2230.1229...]])

if __name__ == "__main__":
	ex1()
