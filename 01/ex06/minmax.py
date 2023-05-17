import numpy as np
from sklearn.preprocessing import MinMaxScaler

def minmax(x):
	"""
	Computes the normalized version of a non-empty numpy.ndarray using the min-max standardization.

	Args:
	x: has to be an numpy.ndarray, a vector.

	Returns:
	x’ as a numpy.ndarray.
	None if x is a non-empty numpy.ndarray or not a numpy.ndarray.

	Raises:
	This function shouldn’t raise any Exception.
	"""
	if not isinstance(x, np.ndarray):
		print("Invalid input: arguments of ndarray type required")	
		return None

	if x.ndim == 1:
		x = x.reshape(x.size, 1)
	elif not (x.ndim == 2 and x.shape[1] == 1):
		print("Invalid input: wrong shape of x", x.shape)
		return None
	return (x - x.min()) / (x.max() - x.min())

def ex1():
	# Example 1:
	X = np.array([0, 15, -9, 7, 12, 3, -21]).reshape((-1, 1))
	ss = MinMaxScaler()
	ss.fit(X)
	print("mine:", minmax(X)) # Output: array([0.58333333, 1. , 0.33333333, 0.77777778, 0.91666667, 0.66666667, 0. ])
	print("real:", ss.transform(X))
	print()
	
	# Example 2:
	Y = np.array([2, 14, -13, 5, 12, 4, -19]).reshape((-1, 1))
	ss.fit(Y)
	print("mine:", minmax(Y)) # Output: array([0.63636364, 1. , 0.18181818, 0.72727273, 0.93939394, 0.6969697 , 0. ])
	print("real:", ss.transform(Y))

if __name__ == "__main__":
	ex1()
