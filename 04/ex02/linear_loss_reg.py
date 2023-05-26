import numpy as np

def reg_loss_(y, y_hat, theta, lambda_):
	"""
	Computes the regularized loss of a linear regression model from two non-empty numpy.array, without any for loop.
	
	Args:
	y: has to be an numpy.ndarray, a vector of shape m * 1.
	y_hat: has to be an numpy.ndarray, a vector of shape m * 1.
	theta: has to be a numpy.ndarray, a vector of shape n * 1.
	lambda_: has to be a float.

	Returns:
	The regularized loss as a float.
	None if y, y_hat, or theta are empty numpy.ndarray.
	None if y and y_hat do not share the same shapes.

	Raises:
	This function should not raise any Exception.
	"""
	for v in [y, y_hat, theta]:
		if not isinstance(v, np.ndarray):
			print(f"Invalid input: argument {v} of ndarray type required")  
			return None

	if y.ndim == 1:
		y = y.reshape(-1, 1)
	elif not (y.ndim == 2 and y.shape[1] == 1):
		print(f"Invalid input: wrong shape of y", y.shape)  
		return None

	if y_hat.ndim == 1:
		y_hat = y_hat.reshape(-1, 1)
	elif not (y_hat.ndim == 2 and y_hat.shape[1] == 1):
		print(f"Invalid input: wrong shape of y_hat", y_hat.shape)  
		return None

	if y.shape[0] != y_hat.shape[0]:
		print(f"Invalid input: unmatched shapes of y and y_hat", y.shape, y_hat.shape)  
		return None
	
	if theta.ndim == 1:
		theta = theta.reshape(-1, 1)
	elif not (theta.ndim == 2 and theta.shape[1] == 1):
		print(f"Invalid input: wrong shape of theta", theta.shape)  
		return None

	if not isinstance(lambda_, float):
		print(f"Invalid input: argument {v} of ndarray type required")  
		return None

	loss = np.dot((y_hat - y).T, (y_hat - y)) + lambda_ * (float(np.sum(theta[1:].T.dot(theta[1:]))))
	return float(loss / (y.shape[0] * 2))

def ex1():	
	y = np.array([2, 14, -13, 5, 12, 4, -19]).reshape((-1, 1))
	y_hat = np.array([3, 13, -11.5, 5, 11, 5, -20]).reshape((-1, 1))
	theta = np.array([1, 2.5, 1.5, -0.9]).reshape((-1, 1))
	# Example :
	print(reg_loss_(y, y_hat, theta, .5)) # Output: 0.8503571428571429
	# Example :
	print(reg_loss_(y, y_hat, theta, .05)) # Output: 0.5511071428571429
	# Example :
	print(reg_loss_(y, y_hat, theta, .9)) # Output: 1.116357142857143

if __name__ == "__main__":
	ex1()
