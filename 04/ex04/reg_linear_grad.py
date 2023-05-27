import numpy as np

def reg_linear_grad(y, x, theta, lambda_):
	"""
	Computes the regularized linear gradient of three non-empty numpy.ndarray,
	with two for-loop. The three arrays must have compatible shapes.

	Args:
	y: has to be a numpy.ndarray, a vector of shape m * 1.
	x: has to be a numpy.ndarray, a matrix of dimesion m * n.
	theta: has to be a numpy.ndarray, a vector of shape (n + 1) * 1.
	lambda_: has to be a float.

	Return:
	A numpy.ndarray, a vector of shape (n + 1) * 1, containing the results of the formula for all j.
	None if y, x, or theta are empty numpy.ndarray.
	None if y, x or theta does not share compatibles shapes.
	None if y, x or theta or lambda_ is not of the expected type.

	Raises:
	This function should not raise any Exception.
	"""
	for v in [x, y, theta]:
		if not isinstance(v, np.ndarray):
			print(f"Invalid input: argument {v} of ndarray type required")  
			return None

	if y.ndim == 1:
		y = y.reshape(-1, 1)
	elif not (y.ndim == 2 and y.shape[1] == 1):
		print(f"Invalid input: wrong shape of y", y.shape)  
		return None

	if not (x.ndim == 2):
		print(f"Invalid input: wrong shape of x", x.shape)  
		return None

	if y.shape[0] != x.shape[0]:
		print(f"Invalid input: unmatched shapes of y and x", y.shape, x.shape)  
		return None
	
	if theta.ndim == 1 and theta.shape[0] == x.shape[1] + 1:
		theta = theta.reshape(x.shape[1] + 1, 1)
	elif not (theta.ndim == 2 and theta.shape[0] == x.shape[1] + 1):
		print(f"Invalid input: wrong shape of theta", theta.shape)  
		return None

	if not isinstance(lambda_, float):
		print(f"Invalid input: argument lambda_ of float type required")  
		return None

	J_elem = []
	gradient = []
	float_sum = 0.0 
	for i in range(x.shape[0]):
		y_hat_row_sum = 0.0
		for j in range(x.shape[1]):
			y_hat_row_sum += theta[j + 1] * x[i][j]
		J_elem.append(float(theta[0] + y_hat_row_sum - y[i]))
		float_sum += J_elem[i]
	gradient.append([float_sum / len(y)])

	for j in range(x.shape[1]):
		float_mul_sum = 0.0
		for elem, i in zip(J_elem, range(x.shape[0])):
			float_mul_sum += float(elem * x[i][j])
		float_mul_sum += float(lambda_ * theta[j + 1])
		gradient.append([float_mul_sum / len(y)])

	return np.array(gradient)

def vec_reg_linear_grad(y, x, theta, lambda_):
	"""
	Computes the regularized linear gradient of three non-empty numpy.ndarray,
	without any for-loop. The three arrays must have compatible shapes.

	Args:
	y: has to be a numpy.ndarray, a vector of shape m * 1.
	x: has to be a numpy.ndarray, a matrix of dimesion m * n.
	theta: has to be a numpy.ndarray, a vector of shape (n + 1) * 1.
	lambda_: has to be a float.

	Return:
	A numpy.ndarray, a vector of shape (n + 1) * 1, containing the results of the formula for all j.
	None if y, x, or theta are empty numpy.ndarray.
	None if y, x or theta does not share compatibles shapes.
	None if y, x or theta or lambda_ is not of the expected type.

	Raises:
	This function should not raise any Exception.
	"""
	for v in [x, y, theta]:
		if not isinstance(v, np.ndarray):
			print(f"Invalid input: argument {v} of ndarray type required")  
			return None

	if y.ndim == 1:
		y = y.reshape(-1, 1)
	elif not (y.ndim == 2 and y.shape[1] == 1):
		print(f"Invalid input: wrong shape of y", y.shape)  
		return None

	if not (x.ndim == 2):
		print(f"Invalid input: wrong shape of x", x.shape)  
		return None

	if y.shape[0] != x.shape[0]:
		print(f"Invalid input: unmatched shapes of y and x", y.shape, x.shape)  
		return None
	
	if theta.ndim == 1 and theta.shape[0] == x.shape[1] + 1:
		theta = theta.reshape(x.shape[1] + 1, 1)
	elif not (theta.ndim == 2 and theta.shape[0] == x.shape[1] + 1):
		print(f"Invalid input: wrong shape of theta", theta.shape)  
		return None

	if not isinstance(lambda_, float):
		print(f"Invalid input: argument lambda_ of float type required")  
		return None

	X = np.hstack((np.ones((x.shape[0], 1)), x))
	theta_ = lambda_ * theta
	theta_[0] = 0
	gradient = np.dot(X.T, (X.dot(theta) - y)) + theta_
	return gradient / x.shape[0]

def ex1():	
	x = np.array([
	[ -6, -7, -9],
	[ 13, -2, 14],
	[ -7, 14, -1],
	[ -8, -4, 6],
	[ -5, -9, 6],
	[ 1, -5, 11],
	[ 9, -11, 8]])
	y = np.array([[2], [14], [-13], [5], [12], [4], [-19]])
	theta = np.array([[7.01], [3], [10.5], [-6]])
	# Example 1.1:
	print(reg_linear_grad(y, x, theta, 1.0)) # Output: array([[ -60.99 ], [-195.64714286], [ 863.46571429], [-644.52142857]])

	# Example 1.2:
	print(vec_reg_linear_grad(y, x, theta, 1.0)) # Output: array([[ -60.99 ], [-195.64714286], [ 863.46571429], [-644.52142857]])

	# Example 2.1:
	print(reg_linear_grad(y, x, theta, 0.5)) # Output: array([[ -60.99 ], [-195.86142857], [ 862.71571429], [-644.09285714]])

	# Example 2.2:
	print(vec_reg_linear_grad(y, x, theta, 0.5)) # Output: array([[ -60.99 ], [-195.86142857], [ 862.71571429], [-644.09285714]])

	# Example 3.1:
	print(reg_linear_grad(y, x, theta, 0.0)) # Output: array([[ -60.99 ], [-196.07571429], [ 861.96571429], [-643.66428571]])

	# Example 3.2:
	print(vec_reg_linear_grad(y, x, theta, 0.0)) # Output: array([[ -60.99 ], [-196.07571429], [ 861.96571429], [-643.66428571]])

if __name__ == "__main__":
	ex1()
