import numpy as np
import sys, os

path = os.path.join(os.path.dirname(__file__), '..', 'ex01')
sys.path.insert(1, path)
from log_pred import logistic_predict_ 

def log_loss_(y, y_hat, eps=1e-15):
	"""
	Computes the logistic loss value.

	Args:
	y: has to be an numpy.ndarray, a vector of shape m * 1.
	y_hat: has to be an numpy.ndarray, a vector of shape m * 1.
	eps: has to be a float, epsilon (default=1e-15)

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
	loss = 0.0
	for i in range(y.shape[0]):
		loss += -(np.sum(y[i] * np.log(y_hat[i])) + (1 - y[i]) * np.log(1 - y_hat[i]))
	return float(loss / len(y))
	
def ex1():
	# Example 1:
	y1 = np.array([1]).reshape((-1, 1))
	x1 = np.array([4]).reshape((-1, 1))
	theta1 = np.array([[2], [0.5]])
	y_hat1 = logistic_predict_(x1, theta1)
	print(log_loss_(y1, y_hat1)) # Output: 0.01814992791780973

	# Example 2:
	y2 = np.array([[1], [0], [1], [0], [1]])
	x2 = np.array([[4], [7.16], [3.2], [9.37], [0.56]])
	theta2 = np.array([[2], [0.5]])
	y_hat2 = logistic_predict_(x2, theta2)
	print(log_loss_(y2, y_hat2)) # Output: 2.4825011602474483

	# Example 3:
	y3 = np.array([[0], [1], [1]])
	x3 = np.array([[0, 2, 3, 4], [2, 4, 5, 5], [1, 3, 2, 7]])
	theta3 = np.array([[-2.4], [-1.5], [0.3], [-1.4], [0.7]])
	y_hat3 = logistic_predict_(x3, theta3)
	print(log_loss_(y3, y_hat3)) # Output: 2.9938533108607053

def ex2():
	y=np.array([[0], [0]])
	y_hat=np.array([[0], [0]]) 
	eps=1e-15	
	print(log_loss_(y, y_hat)) # a value very closed to 0 (something around 1e-15).

	y=np.array([[0], [1]])
	y_hat=np.array([[0], [1]]) 
	eps=1e-15	
	print(log_loss_(y, y_hat)) # a very small value closed to 0 (something around 1e-15).

	y=np.array([[0], [0], [0]])
	y_hat=np.array([[1], [0], [0]])
	eps=1e-15
	print(log_loss_(y, y_hat)) # 11.51292546.

	y=np.array([[0], [0], [0]])
	y_hat=np.array([[1], [0], [1]]) 
	eps=1e-15
	print(log_loss_(y, y_hat)) # 23.02585093.

	y=np.array([[0], [1], [0]])
	y_hat=np.array([[1], [0], [1]])
	eps=1e-15
	print(log_loss_(y, y_hat)) # 34.53877639.

if __name__ == "__main__":
	ex1()
