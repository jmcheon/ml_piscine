import numpy as np
import sys, os

path = os.path.join(os.path.dirname(__file__), '..', 'ex01')
sys.path.insert(1, path)
from log_pred import logistic_predict_ 

def vec_log_loss_(y, y_hat, eps=1e-15):
	"""
	Compute the logistic loss value.

	Args:
	y: has to be an numpy.ndarray, a vector of shape m * 1.
	y_hat: has to be an numpy.ndarray, a vector of shape m * 1.
	eps: epsilon (default=1e-15)

	Returns:
	The logistic loss value as a float.
	None on any error.

	Raises:
	This function should not raise any Exception.
	"""
	for v in [y, y_hat]:
		if not isinstance(v, np.ndarray):
			print(f"Invalid input: argument {v} of ndarray type required")	
			return None

	if not isinstance(eps, float):
		print(f"Invalid input: argument esp of float type required")	
		return None
	
	v = [y, y_hat]
	for i in range(len(v)): 
		if v[i].ndim == 1:
			v[i] = v[i].reshape(v[i].size, 1)
		elif not (v[i].ndim == 2 and v[i].shape[1] == 1):
			print(f"Invalid input: wrong shape of {v[i]}", v[i].shape)
			return None
	y, y_hat = v
	if y.shape != y_hat.shape:
		print(f"Invalid input: two vectors of compatible shape are required")
		return None
	# Clip values to avoid numerical instability
	y_hat = np.clip(y_hat, eps, 1 - eps)  
	loss = -np.mean(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))
	return float(loss)

def ex1():
	# Example 1:
	y1 = np.array([1]).reshape((-1, 1))
	x1 = np.array([4]).reshape((-1, 1))
	theta1 = np.array([[2], [0.5]])
	y_hat1 = logistic_predict_(x1, theta1)
	print(vec_log_loss_(y1, y_hat1)) # Output: 0.018149927917808714

	# Example 2:
	y2 = np.array([[1], [0], [1], [0], [1]])
	x2 = np.array([[4], [7.16], [3.2], [9.37], [0.56]])
	theta2 = np.array([[2], [0.5]])
	y_hat2 = logistic_predict_(x2, theta2)
	print(vec_log_loss_(y2, y_hat2)) # Output: 2.4825011602472347

	# Example 3:
	y3 = np.array([[0], [1], [1]])
	x3 = np.array([[0, 2, 3, 4], [2, 4, 5, 5], [1, 3, 2, 7]])
	theta3 = np.array([[-2.4], [-1.5], [0.3], [-1.4], [0.7]])
	y_hat3 = logistic_predict_(x3, theta3)
	print(vec_log_loss_(y3, y_hat3)) # Output: 2.993853310859968

if __name__ == "__main__":
	ex1()
