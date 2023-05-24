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
	elif not (x.ndim == 2 and x.shape[1] == 1):
		print(f"Invalid input: wrong shape of x", x.shape)
		return None

	if power < 0:
		return None
	elif power == 0:
		return np.ones((x.shape[0], 1))
	elif power == 1:
		return np.copy(x).reshape(-1, 1)
	else:
		return np.vander(x.reshape(-1,), power + 1, increasing=True)[:, 1:]
	

	
def ex1():
	x = np.arange(1,6).reshape(-1, 1)
	print("x:", x, x.shape)

	# Example 0:
	print(add_polynomial_features(x, 3))
	# Example 1:
	print(add_polynomial_features(x, 6))

def ex2():
	x1 = np.arange(1,6).reshape(-1,1)
	x1_poly = add_polynomial_features(x1, 5)
	print(x1_poly) # array([[ 1, 1, 1, 1,1], [ 2, 4, 8, 16, 32], [ 3, 9, 27, 81, 243], [ 4, 16, 64, 256, 1024], [ 5, 25, 125, 625, 3125]])
	
	x2 = np.arange(10,40, 10).reshape(-1,1)
	x2_poly = add_polynomial_features(x2, 5)
	print(x2_poly) # array([[10, 100, 1000, 10000, 100000] [20, 400, 8000, 160000, 3200000] [30, 900, 27000, 810000, 24300000]]])

	x3 = np.arange(10,40, 10).reshape(-1,1)/10
	x3_poly = add_polynomial_features(x3, 3)
	print(x3_poly) # array([[1. , 1. , 1. ], [ 2. , 4. , 8. ], [ 3. , 9. , 27. ]])

if __name__ == "__main__":
	ex2()
