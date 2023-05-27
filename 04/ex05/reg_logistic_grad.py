import numpy as np

def reg_logistic_grad(y, x, theta, lambda_):
	"""
	Computes the regularized logistic gradient of three non-empty numpy.ndarray, with two for-loops.
	
	Args:
	y: has to be a numpy.ndarray, a vector of shape m * 1.
	x: has to be a numpy.ndarray, a matrix of dimesion m * n.
	theta: has to be a numpy.ndarray, a vector of shape n * 1.
	lambda_: has to be a float.

	Returns:
	A numpy.ndarray, a vector of shape n * 1, containing the results of the formula for all j.
	None if y, x, or theta are empty numpy.ndarray.
	None if y, x or theta does not share compatibles shapes.

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
	y_hat = np.array(1 / (1 + np.exp(-X.dot(theta))))
	J_elem = (y_hat - y)

	gradient = []
	float_sum = 0.0 
	for elem in J_elem:
		float_sum += float(elem)
	gradient.append([float_sum / len(y)])

	for j in range(x.shape[1]):
		float_mul_sum = 0.0
		for elem, i in zip(J_elem, range(x.shape[0])):
			float_mul_sum += float(elem * x[i][j])
		float_mul_sum += float(lambda_ * theta[j + 1])
		gradient.append([float_mul_sum / len(y)])

	return np.array(gradient)

def vec_reg_logistic_grad(y, x, theta, lambda_):
	"""
	Computes the regularized logistic gradient of three non-empty numpy.ndarray, without any for-loop.
	
	Args:
	y: has to be a numpy.ndarray, a vector of shape m * 1.
	x: has to be a numpy.ndarray, a matrix of shape m * n.
	theta: has to be a numpy.ndarray, a vector of shape n * 1.
	lambda_: has to be a float.

	Returns:
	A numpy.ndarray, a vector of shape n * 1, containing the results of the formula for all j.
	None if y, x, or theta are empty numpy.ndarray.
	None if y, x or theta does not share compatibles shapes.

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
	y_hat = np.array(1 / (1 + np.exp(-X.dot(theta))))
	theta_ = lambda_ * theta
	theta_[0] = 0
	gradient = np.dot(X.T, (y_hat - y)) + theta_
	return gradient / x.shape[0]

def ex1():
	x = np.array([[0, 2, 3, 4],
	[2, 4, 5, 5],
	[1, 3, 2, 7]])
	y = np.array([[0], [1], [1]])
	theta = np.array([[-2.4], [-1.5], [0.3], [-1.4], [0.7]])

	# Example 1.1:
	print(reg_logistic_grad(y, x, theta, 1.0)) # Output: array([[-0.55711039], [-1.40334809], [-1.91756886], [-2.56737958], [-3.03924017]])

	# Example 1.2:
	print(vec_reg_logistic_grad(y, x, theta, 1.0)) # Output: array([[-0.55711039], [-1.40334809], [-1.91756886], [-2.56737958], [-3.03924017]])

	# Example 2.1:
	print(reg_logistic_grad(y, x, theta, 0.5)) # Output: array([[-0.55711039], [-1.15334809], [-1.96756886], [-2.33404624], [-3.15590684]])

	# Example 2.2:
	print(vec_reg_logistic_grad(y, x, theta, 0.5)) # Output: array([[-0.55711039], [-1.15334809], [-1.96756886], [-2.33404624], [-3.15590684]])

	# Example 3.1:
	print(reg_logistic_grad(y, x, theta, 0.0)) # Output: array([[-0.55711039], [-0.90334809], [-2.01756886], [-2.10071291], [-3.27257351]])

	# Example 3.2:
	print(vec_reg_logistic_grad(y, x, theta, 0.0)) # Output: array([[-0.55711039], [-0.90334809], [-2.01756886], [-2.10071291], [-3.27257351]])

if __name__ == "__main__":
	ex1()
