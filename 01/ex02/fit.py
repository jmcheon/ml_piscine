import numpy as np

def fit_(x, y, theta, alpha, max_iter):
	"""
	Description:
	Fits the model to the training dataset contained in x and y.

	Args:
	x: has to be a numpy.ndarray, a vector of dimension m * 1: (number of training examples, 1).
	y: has to be a numpy.ndarray, a vector of dimension m * 1: (number of training examples, 1).
	theta: has to be a numpy.ndarray, a vector of dimension 2 * 1.
	alpha: has to be a float, the learning rate
	max_iter: has to be an int, the number of iterations done during the gradient descent

	Returns:
	new_theta: numpy.ndarray, a vector of dimension 2 * 1.
	None if there is a matching dimension problem.

	Raises:
	This function should not raise any Exception.
	"""
	# arguments type varification
	for v in [x, y, theta]:
		if not isinstance(v, np.ndarray):
			print(f"Invalid input: argument {v} of ndarray type required")	
			return None
	if not isinstance(alpha, float):
		print(f"Invalid input: argument alpha of float type required")	
		return None
	if not isinstance(max_iter, int):
		print(f"Invalid input: argument max_iter of int type required")	
		return None

	# vector arguments shape varification
	v = [x, y]
	for i in range(len(v)): 
		if v[i].ndim == 1:
			v[i] = v[i].reshape(v[i].size, 1)
		elif not (v[i].ndim == 2 and v[i].shape[1] == 1):
			print(f"Invalid input: wrong shape of {v[i]}", v[i].shape)
			return None
	if theta.ndim == 1:
		theta = theta.reshape(theta.size, 1)
	elif not (theta.ndim == 2 and theta.shape == (2, 1)):
		print(f"Invalid input: wrong shape of {theta}", theta.shape)
		return None

	# We add a column of 1's for the column of interception
	X = np.hstack((np.ones((x.shape[0], 1)), x))
	#print("X:", X, X.shape)
	X_t = np.transpose(X)
	#print("X_t:", X_t, X_t.shape)

	gradient = X_t.dot(X.dot(theta) - y) / len(y)
	
	for i in max_iter:

	return new_theta
	
def ex1():
	x = np.array([[12.4956442], [21.5007972], [31.5527382], [48.9145838], [57.5088733]])
	y = np.array([[37.4013816], [36.1473236], [45.7655287], [46.6793434], [59.5585554]])
	theta= np.array([1, 1]).reshape((-1, 1))
	# Example 0:
	theta1 = fit_(x, y, theta, alpha=5e-8, max_iter=1500000)
	print(theta1)
	
	# Example 1:
	print(predict(x, theta1))

if __name__ == "__main__":
	ex1()
