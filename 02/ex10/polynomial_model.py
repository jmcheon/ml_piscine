import numpy as np

def add_polynomial_features(x, power):
	"""
	Add polynomial features to vector x by raising its values up to the power given in argument.

	Args:
	x: has to be an numpy.array, a vector of dimension m * 1.
	power: has to be an int, the power up to which the components of vector x are going to be raised.

	Return:
	The matrix of polynomial features as a numpy.array, of dimension m * n,
	containing the polynomial feature values for all training examples.
	None if x is an empty numpy.array.
	None if x or power is not of expected type.

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
		x = x.reshape(x.size, 1)
	elif not (x.ndim == 2):
		print(f"Invalid input: wrong shape of x", x.shape)
		return None

	result = x.copy()
	for i in range(power - 1):
		result = np.hstack(
		(result, np.power(result[:, 0], i + 2).reshape(-1, 1)))
	return result

	if power < 0:
		return None
	elif power == 0:
		return np.ones((x.shape[0], 1))
	elif power == 1:
		return np.copy(x).reshape(-1, 1)
	else:
		print("power:", power)
		return np.vander(x.reshape(-1,), power + 1, increasing=True)[:, 1:]
	

	
def ex1():
	x = np.arange(1,6).reshape(-1, 1)
	print("x:", x, x.shape)

	# Example 0:
	print(add_polynomial_features(x, 3))
	# Example 1:
	print(add_polynomial_features(x, 6))

if __name__ == "__main__":
	ex1()
