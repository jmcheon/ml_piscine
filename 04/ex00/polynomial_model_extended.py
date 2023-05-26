import numpy as np

def add_polynomial_features(x, power):
	"""
	Add polynomial features to matrix x by raising its columns to every power in the range of 1 up to the power give

	Args:
	x: has to be an numpy.ndarray, a matrix of shape m * n.
	power: has to be an int, the power up to which the columns of matrix x are going to be raised.

	Returns:
	The matrix of polynomial features as a numpy.ndarray, of shape m * (np), containg the polynomial feature vaNone if x is an empty numpy.ndarray.

	Raises:
	This function should not raise any Exception.
	"""
	if not isinstance(power, int):
		print(f"Invalid input: argument power of int type required")  
		return None 
	if not isinstance(x, np.ndarray):
		print(f"Invalid input: argument x of ndarray type required")  
		return None
	
	if x.ndim == 1:
		x = x.reshape(-1, 1)
	elif not x.ndim == 2:
		print(f"Invalid input: wrong shape of x", x.shape)  
		return None
	
	if power < 0:
		return None
	elif power == 0:
		return np.ones((x.shape[0], 1))
	elif power == 1:
		return np.copy(x)
	else:
		x_poly = np.ones((x.shape[0], x.shape[1] * power))
		for i in range(power):
			x_poly[:, i * x.shape[1] : (i + 1) * x.shape[1]] = x ** (i + 1)
		return x_poly
	
def ex1():
	x = np.arange(1,11).reshape(5, 2)
	# Example 1:
	print(add_polynomial_features(x, 3))
	# Example 2:
	print(add_polynomial_features(x, 4))

if __name__ == "__main__":
	ex1()
