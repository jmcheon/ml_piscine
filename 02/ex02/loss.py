import numpy as np

def loss_(y, y_hat):
	"""
	Computes the mean squared error of two non-empty numpy.array, without any for loop.
	The two arrays must have the same dimensions.

	Args:
	y: has to be an numpy.array, a vector.
	y_hat: has to be an numpy.array, a vector.

	Return:
	The mean squared error of the two vectors as a float.
	None if y or y_hat are empty numpy.array.
	None if y and y_hat does not share the same dimensions.
	None if y or y_hat is not of expected type.

	Raises:
	This function should not raise any Exception.
	"""
	for v in [y, y_hat]:
		if not isinstance(v, np.ndarray):
			print(f"Invalid input: argument {v} of ndarray type required")	
			return None

	for v in [y, y_hat]:
		if not (v.ndim == 2 and v.shape in [(v.size, 1), (1, v.size)]):
			print(f"Invalid input: wrong shape of {v}", v.shape)
			return None

	if y.shape != y_hat.shape:
		print(f"Invalid input: shape of vectors should be compatiable.")
		return None

	squared_errors = (y_hat - y) ** 2
	return np.sum(squared_errors) / (2 * len(y))
	
def ex1():	
	X = np.array([0, 15, -9, 7, 12, 3, -21]).reshape((-1, 1))
	Y = np.array([2, 14, -13, 5, 12, 4, -19]).reshape((-1, 1))
	#print("X:", X, X.shape)
	#print("Y:", Y, Y.shape)
	
	# Example 1:
	print(loss_(X, Y)) # Output: 2.142857142857143
	
	# Example 2:
	print(loss_(X, X)) # Output: 0.0

def ex2():
	n = 1
	y = (np.ones(n)).reshape(-1,1)
	y_hat = (np.zeros(n)).reshape(-1,1)
	print(loss_(y, y_hat)) # Output: 0.5 

	y = (np.ones(n)).reshape(-1,1)+4
	y_hat = (np.zeros(n)).reshape(-1,1)
	print(loss_(y, y_hat)) # Output: 12.5

	y = (np.ones(7)).reshape(-1,1)+4
	y_hat = (np.arange(7)).reshape(-1,1)
	print(loss_(y, y_hat)) # Output: 4

if __name__ == "__main__":
	ex2()
