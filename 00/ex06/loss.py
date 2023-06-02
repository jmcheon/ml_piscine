import numpy as np
import sys, os

path = os.path.join(os.path.dirname(__file__), '..', 'ex04')
sys.path.insert(1, path)
from prediction import predict_

def loss_elem_(y, y_hat):
	"""
	Description:
	Calculates all the elements (y_pred - y)^2 of the loss function.

	Args:
	y: has to be an numpy.array, a vector.
	y_hat: has to be an numpy.array, a vector.

	Returns:
	J_elem: numpy.array, a vector of dimension (number of the training examples,1).
	None if there is a dimension matching problem between X, Y or theta.
	None if any argument is not of the expected type.

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

	return (y_hat - y) ** 2
	
def loss_(y, y_hat):
	"""
	Description:
	Calculates the value of loss function.

	Args:
	y: has to be an numpy.array, a vector.
	y_hat: has to be an numpy.array, a vector.

	Returns:
	J_value : has to be a float.
	None if there is a dimension matching problem between X, Y or theta.
	None if any argument is not of the expected type.

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

	J_elem = loss_elem_(y, y_hat)
	float_sum = float(np.sum(J_elem))
	return float_sum / (2 * len(y))

def ex1():
	x1 = np.array([[0.], [1.], [2.], [3.], [4.]])
	theta1 = np.array([[2.], [4.]])
	y_hat1 = predict_(x1, theta1)
	# print("y_hat1:", y_hat1, y_hat1.shape)

	y1 = np.array([[2.], [7.], [12.], [17.], [22.]])
	#ret = [float(y_hat1_i - y1_i) for y_hat1_i, y1_i in zip(y_hat1, y1)]
	#print("y_hat1 - y1:", ret) 
	#print("(y_hat1 - y1)^2:", np.array(ret) ** 2) 

	
	# Example 1:
	print(loss_elem_(y1, y_hat1)) # Output: array([[0.], [1], [4], [9], [16]])
	# Example 2:
	print(loss_(y1, y_hat1)) # Output: 3.0


	x2 = np.array([0, 15, -9, 7, 12, 3, -21]).reshape(-1, 1)
	theta2 = np.array([[0.], [1.]]).reshape(-1, 1)
	y_hat2 = predict_(x2, theta2)
	y2 = np.array([2, 14, -13, 5, 12, 4, -19]).reshape(-1, 1)
	
	# Example 3:
	print(loss_(y2, y_hat2)) # Output: 2.142857142857143
	# Example 4:
	print(loss_(y2, y2)) # Output: 0.0

	y_hat = np.array([[1], [2], [3], [4]])
	y = np.array([[0], [0], [0], [0]])
	#ret = [float(y_hat_i - y_i) for y_hat_i, y_i in zip(y_hat, y)]
	#print("y_hat - y:", ret) 
	#print("(y_hat - y)^2:", np.array(ret) ** 2) 
	#print(loss_elem_(y, y_hat)) # Output: 

if __name__ == "__main__":
	ex1()
