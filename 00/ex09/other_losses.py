import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from math import sqrt

def mse_(y, y_hat):
	"""
	Description:
	Calculate the MSE between the predicted output and the real output.

	Args:
	y: has to be a numpy.array, a vector of dimension m * 1.
	y_hat: has to be a numpy.array, a vector of dimension m * 1.

	Returns:
	mse: has to be a float.
	None if there is a matching dimension problem.

	Raises:
	This function should not raise any Exceptions.
	"""
	for v in [y, y_hat]:
		if not isinstance(v, np.ndarray):
			print(f"Invalid input: argument {v} of ndarray type required")	
			return None

	v = [y, y_hat]
	for i in range(len(v)): 
		if v[i].ndim == 1:
			v[i] = v[i].reshape(v[i].size, 1)
		elif not (v[i].ndim == 2 and v[i].shape[1] == 1):
			print(f"Invalid input: wrong shape of {v[i]}", v[i].shape)
			return None

	J_elem = (y_hat - y) ** 2
	float_sum = float(np.sum(J_elem))
	mse = float_sum / len(y)
	return mse 

def rmse_(y, y_hat):
	"""
	Description:
	Calculate the RMSE between the predicted output and the real output.

	Args:
	y: has to be a numpy.array, a vector of dimension m * 1.
	y_hat: has to be a numpy.array, a vector of dimension m * 1.

	Returns:
	rmse: has to be a float.
	None if there is a matching dimension problem.

	Raises:
	This function should not raise any Exceptions.
	"""
	for v in [y, y_hat]:
		if not isinstance(v, np.ndarray):
			print(f"Invalid input: argument {v} of ndarray type required")	
			return None

	v = [y, y_hat]
	for i in range(len(v)): 
		if v[i].ndim == 1:
			v[i] = v[i].reshape(v[i].size, 1)
		elif not (v[i].ndim == 2 and v[i].shape[1] == 1):
			print(f"Invalid input: wrong shape of {v[i]}", v[i].shape)
			return None
	mse = mse_(y, y_hat) 
	rmse = mse ** 0.5
	return rmse

def mae_(y, y_hat):
	"""
	Description:
	Calculate the MAE between the predicted output and the real output.

	Args:
	y: has to be a numpy.array, a vector of dimension m * 1.
	y_hat: has to be a numpy.array, a vector of dimension m * 1.

	Returns:
	mae: has to be a float.
	None if there is a matching dimension problem.

	Raises:
	This function should not raise any Exceptions.
	"""
	for v in [y, y_hat]:
		if not isinstance(v, np.ndarray):
			print(f"Invalid input: argument {v} of ndarray type required")	
			return None

	v = [y, y_hat]
	for i in range(len(v)): 
		if v[i].ndim == 1:
			v[i] = v[i].reshape(v[i].size, 1)
		elif not (v[i].ndim == 2 and v[i].shape[1] == 1):
			print(f"Invalid input: wrong shape of {v[i]}", v[i].shape)
			return None

	J_elem = np.abs(y_hat - y)
	float_sum = float(np.sum(J_elem))
	mae = float_sum / len(y)
	return mae

def r2score_(y, y_hat):
	"""
	Description:
	Calculate the R2score between the predicted output and the output.

	Args:
	y: has to be a numpy.array, a vector of dimension m * 1.
	y_hat: has to be a numpy.array, a vector of dimension m * 1.

	Returns:
	r2score: has to be a float.
	None if there is a matching dimension problem.

	Raises:
	This function should not raise any Exceptions.
	"""
	for v in [y, y_hat]:
		if not isinstance(v, np.ndarray):
			print(f"Invalid input: argument {v} of ndarray type required")	
			return None

	v = [y, y_hat]
	for i in range(len(v)): 
		if v[i].ndim == 1:
			v[i] = v[i].reshape(v[i].size, 1)
		elif not (v[i].ndim == 2 and v[i].shape[1] == 1):
			print(f"Invalid input: wrong shape of {v[i]}", v[i].shape)
			return None

	J_elem = (y_hat - y) ** 2
	M_elem = (y - np.mean(y)) ** 2
	float_J_sum = float(np.sum(J_elem))
	float_M_sum = float(np.sum(M_elem))
	r2score = 1 - (float_J_sum / float_M_sum)
	return r2score
	
def ex1():	
	# Example 1:
	x = np.array([0, 15, -9, 7, 12, 3, -21])
	y = np.array([2, 14, -13, 5, 12, 4, -19])

	# Mean squared error
	## your implementation
	print("\nmy mse:", mse_(x,y)) ## Output: 4.285714285714286

	## sklearn implementation
	print("sklearn mse:", mean_squared_error(x,y)) ## Output: 4.285714285714286

	# Root mean squared error
	## your implementation
	print("\nmy rmse:", rmse_(x,y)) ## Output: 2.0701966780270626

	## sklearn implementation not available: take the square root of MSE
	print("sklearn rmse:", sqrt(mean_squared_error(x,y))) ## Output: 2.0701966780270626

	# Mean absolute error
	## your implementation
	print("\nmy mae:", mae_(x,y)) # Output: 1.7142857142857142

	## sklearn implementation
	print("sklearn mae:", mean_absolute_error(x,y)) # Output: 1.7142857142857142

	# R2-score
	## your implementation
	print("\nmy r2score:", r2score_(x,y)) ## Output: 0.9681721733858745
	
	## sklearn implementation
	print("sklearn r2score:", r2_score(x,y)) ## Output: 0.9681721733858745

if __name__ == "__main__":
	ex1()
