import numpy as np

def iterative_l2(theta):
	"""
	Computes the L2 regularization of a non-empty numpy.ndarray, with a for-loop.

	Args:
	theta: has to be a numpy.ndarray, a vector of shape n * 1.

	Returns:
	The L2 regularization as a float.
	None if theta in an empty numpy.ndarray.

	Raises:
	This function should not raise any Exception.
	"""
	if not isinstance(theta, np.ndarray):
		print(f"Invalid input: argument theta of ndarray type required")  
		return None
	
	if theta.ndim == 1:
		theta = theta.reshape(-1, 1)
	elif not (theta.ndim == 2 and theta.shape[1] == 1):
		print(f"Invalid input: wrong shape of theta", theta.shape)  
		return None

	l2 = 0.0
	for theta_j in theta[1:]:
		l2 += (theta_j ** 2)
	return float(l2)
	


def l2(theta):
	"""
	Computes the L2 regularization of a non-empty numpy.ndarray, without any for-loop.
	
	Args:
	theta: has to be a numpy.ndarray, a vector of shape n * 1.

	Returns:
	The L2 regularization as a float.
	None if theta in an empty numpy.ndarray.

	Raises:
	This function should not raise any Exception.
	"""
	if not isinstance(theta, np.ndarray):
		print(f"Invalid input: argument theta of ndarray type required")  
		return None
	
	if theta.ndim == 1:
		theta = theta.reshape(-1, 1)
	elif not (theta.ndim == 2 and theta.shape[1] == 1):
		print(f"Invalid input: wrong shape of theta", theta.shape)  
		return None
	
	return float(np.sum(theta[1:].transpose().dot(theta[1:])))

def ex1():
	x = np.array([2, 14, -13, 5, 12, 4, -19]).reshape((-1, 1))
	print("x:", x, x.shape)
	# Example 1:
	print(iterative_l2(x)) # Output: 911.0
	# Example 2:
	print(l2(x)) # Output: 911.0
	y = np.array([3,0.5,-6]).reshape((-1, 1))
	# Example 3:
	print(iterative_l2(y)) # Output: 36.25
	# Example 4:
	print(l2(y)) # Output: 36.25

if __name__ == "__main__":
	ex1()
