import numpy as np

def simple_predict(x, theta):
	"""
	Computes the prediction vector y_hat from two non-empty numpy.array.

	Args:
	x: has to be an numpy.array, a matrix of dimension m * n.
	theta: has to be an numpy.array, a vector of dimension (n + 1) * 1.

	Return:
	y_hat as a numpy.array, a vector of dimension m * 1.
	None if x or theta are empty numpy.array.
	None if x or theta dimensions are not matching.
	None if x or theta is not of expected type.

	Raises:
	This function should not raise any Exception.
	"""
	if not isinstance(x, np.ndarray) or not isinstance(theta, np.ndarray):
		print("Invalid input: arguments of ndarray type required")	
		return None

	if not x.ndim == 2:
		print("Invalid input: wrong shape of x", x.shape)
		return None

	if theta.ndim == 1 and theta.size == x.shape[1] + 1:
		theta = theta.reshape(x.shape[1] + 1, 1)
	elif not (theta.ndim == 2 and theta.shape == (x.shape[1] + 1, 1)):
		print("Invalid input: wrong shape of theta ", theta.shape)
		return None
	
	y_hat = []	
	for i in range(x.shape[0]):
		y_hat_row_sum = 0.0
		for j in range(x.shape[1]):
			y_hat_row_sum += theta[j + 1] * x[i][j]
		y_hat.append([float(theta[0] + y_hat_row_sum)])
	return np.array(y_hat)

def ex1():
	x = np.arange(1,13).reshape((4,-1))
	print("x:", x, x.shape)

	# Example 1:
	theta1 = np.array([5, 0, 0, 0]).reshape((-1, 1))
	print("\nexample1:")
	print(simple_predict(x, theta1)) # Ouput: array([[5.], [5.], [5.], [5.]])
	# Do you understand why y_hat contains only 5’s here?
	
	# Example 2:
	theta2 = np.array([0, 1, 0, 0]).reshape((-1, 1))
	print("\nexample2:")
	print(simple_predict(x, theta2)) # Output: array([[ 1.], [ 4.], [ 7.], [10.]])
	# Do you understand why y_hat == x[:,0] here?
	
	# Example 3:
	theta3 = np.array([-1.5, 0.6, 2.3, 1.98]).reshape((-1, 1))
	print("\nexample3:")
	print(simple_predict(x, theta3)) # Output: array([[ 9.64], [24.28], [38.92], [53.56]])
	
	# Example 4:
	theta4 = np.array([-3, 1, 2, 3.5]).reshape((-1, 1))
	print("\nexample4:")
	print(simple_predict(x, theta4)) # Output: array([[12.5], [32. ], [51.5], [71. ]])

def ex2():
	x = np.arange(1,13).reshape(-1,2)
	theta = np.ones(3).reshape(-1,1)
	print(simple_predict(x, theta)) # Ouput: array([[4.], [ 8.], [12.], [16.], [20.], [24.]])

	x = (np.arange(1,13)).reshape(-1,3)
	theta = np.ones(4).reshape(-1,1)
	print(simple_predict(x, theta)) # Ouput: array([[ 7.], [16.], [25.], [34.]])

	x = (np.arange(1,13)).reshape(-1,4) 
	theta = np.ones(5).reshape(-1,1)
	print(simple_predict(x, theta)) # Ouput: array([[11.], [27.], [43.]])

if __name__ == "__main__":
	ex2()
