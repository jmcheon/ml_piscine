import numpy as np

def sigmoid_(x):
	"""
	Compute the sigmoid of a vector.

	Args:
	x: has to be a numpy.ndarray of shape (m, 1).

	Returns:
	The sigmoid value as a numpy.ndarray of shape (m, 1).
	None if x is an empty numpy.ndarray.

	Raises:
	This function should not raise any Exception.
	"""
	if not isinstance(x, np.ndarray):
		print(f"Invalid input: argument x of ndarray type required")	
		return None

	if x.ndim == 1:
		x = x.reshape(x.size, 1)
	elif not (x.ndim == 2 and x.shape[1] == 1):
		print(f"Invalid input: wrong shape of x", x.shape)
		return None
	return np.array(1 / (1 + np.exp(-x)))

def ex1():
	# Example 1:
	x = np.array([[-4]])
	print(sigmoid_(x)) # Output: array([[0.01798620996209156]])
	# Example 2:
	x = np.array([[2]])
	print(sigmoid_(x)) # Output: array([[0.8807970779778823]])
	# Example 3:
	x = np.array([[-4], [2], [0]])
	print(sigmoid_(x)) # Output: array([[0.01798620996209156], [0.8807970779778823], [0.5]])

if __name__ == "__main__":
	ex1()
