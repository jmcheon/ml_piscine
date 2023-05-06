import numpy as np

def simple_gradient(x, y, theta):
	"""
	Computes a gradient vector from three non-empty numpy.array, with a for-loop.
	The three arrays must have compatible shapes.

	Args:
	x: has to be an numpy.array, a vector of shape m * 1.
	y: has to be an numpy.array, a vector of shape m * 1.
	theta: has to be an numpy.array, a 2 * 1 vector.

	Return:
	The gradient as a numpy.array, a vector of shape 2 * 1.
	None if x, y, or theta are empty numpy.array.
	None if x, y and theta do not have compatible shapes.
	None if x, y or theta is not of the expected type.

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

	if theta.ndim == 1:
		theta = theta.reshape(theta.size, 1)
	elif not (theta.ndim == 2 and theta.shape == (2, 1)):
		print(f"Invalid input: wrong shape of {theta}", theta.shape)
		return None

	y_hat = theta[0] + theta[1]*x
	J_elem = (y_hat - y)
	#print("J_elem:", J_elem)
	float_sum = float(np.sum(J_elem))
	float_mul_sum = float(np.sum(J_elem*x))

	j_0 = float_sum / len(y)
	j_1 = float_mul_sum / len(y)
	gradient = np.array([[j_0], [j_1]])
	return gradient 

def ex1():
	x = np.array([12.4956442, 21.5007972, 31.5527382, 48.9145838, 57.5088733]).reshape((-1, 1))
	y = np.array([37.4013816, 36.1473236, 45.7655287, 46.6793434, 59.5585554]).reshape((-1, 1))
	# Example 0:
	theta1 = np.array([2, 0.7]).reshape((-1, 1))
	print(simple_gradient(x, y, theta1)) # Output: array([[-19.0342574], [-586.66875564]])
	
	# Example 1:
	theta2 = np.array([1, -0.4]).reshape((-1, 1))
	print(simple_gradient(x, y, theta2)) # Output: array([[-57.86823748], [-2230.12297889]])

if __name__ == "__main__":
	ex1()
