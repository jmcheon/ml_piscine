import numpy as np

def add_intercept(x):
	"""
	Adds a column of 1’s to the non-empty numpy.array x.

	Args:
	x: has to be a numpy.array of dimension m * n.

	Returns:
	X, a numpy.array of dimension m * (n + 1).
	None if x is not a numpy.array.
	None if x is an empty numpy.array.

	Raises:
	This function should not raise any Exception.
	"""
	if not isinstance(x, np.ndarray):
		print("Invalid input: argument of ndarray type required")	
		return None

	if x.ndim == 1:
		x = x.reshape(x.size, 1)
	elif not x.ndim == 2:
		print("Invalid input: wrong shape of x", x.shape)
		return None

	X = []
	for x_row in x:
		row = [1.0]
		for elem in x_row:
			row.append(elem)
		X.append(row)
	return np.array(X)

def ex1():
	# Example 1:
	x = np.arange(1,6)
	print("x:", x, x.shape)
	print(add_intercept(x))

	# Example 2:
	y = np.arange(1,10).reshape((3,3))
	print("y:", y, y.shape)
	print(add_intercept(y))

if __name__ == "__main__":
	ex1()
