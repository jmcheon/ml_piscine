import numpy as np

def data_spliter(x, y, proportion):
	"""
	Shuffles and splits the dataset (given by x and y) into a training and a test set,
	while respecting the given proportion of examples to be kept in the training set.

	Args:
	x: has to be an numpy.array, a matrix of dimension m * n.
	y: has to be an numpy.array, a vector of dimension m * 1.
	proportion: has to be a float, the proportion of the dataset that will be assigned to the
	training set.

	Return:
	(x_train, x_test, y_train, y_test) as a tuple of numpy.array
	None if x or y is an empty numpy.array.
	None if x and y do not share compatible dimensions.
	None if x, y or proportion is not of expected type.

	Raises:
	This function should not raise any Exception.
	"""
	for v in [x, y]:
		if not isinstance(v, np.ndarray):
			print(f"Invalid input: argument {v} of ndarray type required")	
			return None

	if not x.ndim == 2:
		print(f"Invalid input: wrong shape of x", x.shape)
		return None

	if y.ndim == 1 and y.shape[0] == x.shape[0]:
		y = y.reshape(y.size, 1)
	elif not (y.ndim == 2 and y.shape == (x.shape[0], 1)):
		print(f"Invalid input: wrong shape of y", y.shape)
		return None

	if not isinstance(proportion, float):
		print(f"Invalid input: argument proportion of float type required")	
		return None
	
	data = np.hstack((x, y))
	p = int(x.shape[0] * proportion)
	#np.random.shuffle(data)
	np.random.default_rng(42).shuffle(data)
	x_train, x_test= data[p:, :-1], data[:p, :-1]
	y_train, y_test = data[p:, -1:], data[:p, -1:] 
	return (x_train, x_test, y_train, y_test)


def ex1():
	x1 = np.array([1, 42, 300, 10, 59]).reshape((-1, 1))
	y = np.array([0, 1, 0, 1, 0]).reshape((-1, 1))
	print("x1:", x1, x1.shape)
	print("y:", y, y.shape)

	# Example 1:
	print(data_spliter(x1, y, 0.8))
	# Example 2:
	print(data_spliter(x1, y, 0.5))

def ex2():
	x2 = np.array([[ 1, 42],
			[300, 10],
			[ 59, 1],
			[300, 59],
			[ 10, 42]])
	y = np.array([0, 1, 0, 1, 0]).reshape((-1, 1))
	print("x2:", x2, x2.shape)
	print("y:", y, y.shape)
	# Example 3:
	print(data_spliter(x2, y, 0.8))
	# Example 4:
	print(data_spliter(x2, y, 0.5))

if __name__ == "__main__":
	ex1()
