import numpy as np

class MyLinearRegression():
	"""
	Description: My personnal linear regression class to fit like a boss.
	Methodes: fit_(self, x, y), predict_(self, x), loss_elem_(self, y, y_hat), loss_(self, y, y_hat).
	"""

	def __init__(self, thetas, alpha=0.001, max_iter=1000):
		if self.check_validation(thetas, alpha, max_iter):
			self.alpha = alpha
			self.max_iter = max_iter
			if thetas.ndim == 1:
				self.thetas = thetas.reshape(thetas.size, 1)
			else:
				self.thetas = thetas

	@staticmethod
	def check_validation(thetas, alpha, max_iter):
		if not isinstance(thetas, np.ndarray):
			print(f"Invalid input: argument theta of ndarray type required")	
			return False

		if not isinstance(alpha, float):
			print(f"Invalid input: argument alpha of float type required")	
			return False

		if not isinstance(max_iter, int):
			print(f"Invalid input: argument max_iter of int type required")	
			return False 
		return True

	@staticmethod
	def mse_(y, y_hat):
		"""
		Description: Calculate the MSE between the predicted output and the real output.
		"""
		for v in [y, y_hat]:
			if not isinstance(v, np.ndarray):
				print(f"Invalid input: argument {v} of ndarray type required")	
				return None
	
		v = [y, y_hat]
		for i in range(len(v)): 
			if v[i].ndim == 1:
				v[i] = v[i].reshape(v[i].size, 1)
			elif not (v[i].ndim == 2):
				print(f"Invalid input: wrong shape of {v[i]}", v[i].shape)
				return None
		y, y_hat = v
	
		squared_diff = np.square(y_hat - y)
		mse = np.sum(squared_diff) / len(y)
		# J_elem = (y_hat - y) ** 2
		# float_sum = float(np.sum(J_elem))
		# mse = float_sum / len(y)
		return mse 
	
	@staticmethod
	def gradient(x, y, theta):
		"""
		Computes a gradient vector from three non-empty numpy.array, without any for-loop.
		The three arrays must have the compatible dimensions.

		Args:
		x: has to be an numpy.array, a matrix of dimension m * n.
		y: has to be an numpy.array, a vector of dimension m * 1.
		theta: has to be an numpy.array, a vector (n +1) * 1.

		Return:
		The gradient as a numpy.array, a vector of dimensions n * 1,
		containg the result of the formula for all j.
		None if x, y, or theta are empty numpy.array.
		None if x, y and theta do not have compatible dimensions.
		None if x, y or theta is not of expected type.

		Raises:
		This function should not raise any Exception.
		"""
		for v in [x, y, theta]:
			if not isinstance(v, np.ndarray):
				print(f"Invalid input: argument {v} of ndarray type required")	
				return None

		if not x.ndim == 2:
			print(f"Invalid input: wrong shape of x", x.shape)
			return None

		if y.ndim == 1:
			y = y.reshape(y.size, 1)
		elif not (y.ndim == 2 and y.shape[1] == 1):
			print(f"Invalid input: wrong shape of y", y.shape)
			return None

		if theta.ndim == 1 and theta.size == x.shape[1] + 1:
			theta = theta.reshape(x.shape[1] + 1, 1)
		elif not (theta.ndim == 2 and theta.shape == (x.shape[1] + 1, 1)):
			print(f"Invalid input: wrong shape of {theta}", theta.shape)
			return None
			
		X = np.hstack((np.ones((x.shape[0], 1)), x))
		X_t = np.transpose(X)
		gradient = X_t.dot(X.dot(theta) - y) / len(y)
		return gradient 
	
	def fit_(self, x, y):
		"""
		Description: Fits the model to the training dataset contained in x and y.
		"""
		# arguments type varification
		for v in [x, y]:
			if not isinstance(v, np.ndarray):
				print(f"Invalid input: argument {v} of ndarray type required")
				return None

		if not x.ndim == 2:
			print(f"Invalid input: wrong shape of x", x.shape)
			return None

		if y.ndim == 1:
			y = y.reshape(y.size, 1)
		elif not (y.ndim == 2 and y.shape[1] == 1):
			print(f"Invalid input: wrong shape of y", y.shape)
			return None

		if self.thetas.ndim == 1 and self.thetas.size == x.shape[1] + 1:
			self.thetas = self.thetas.reshape(x.shape[1] + 1, 1)
		elif not (self.thetas.ndim == 2 and self.thetas.shape == (x.shape[1] + 1, 1)):
			print(f"Invalid input: wrong shape of {self.thetas}", self.thetas.shape)
			return None
		# Weights to update: alpha * mean((y_hat - y) * x) 
		# Bias to update: alpha * mean(y_hat - y)
		new_theta = np.copy(self.thetas.astype("float64"))
		for _ in range(self.max_iter):
			# Compute gradient descent
			grad = self.gradient(x, y ,new_theta)
   		        # Handle invalid values in the gradient
			if np.any(np.isnan(grad)) or np.any(np.isinf(grad)):
				#print("Warning: Invalid values encountered in the gradient. Skipping update.")
				continue
			# Update new_theta
			new_theta -= (self.alpha * grad)
		self.thetas = new_theta
		return self.thetas

	def predict_(self, x):
		"""
		Description: Computes the vector of prediction y_hat from two non-empty numpy.array.
		"""
		if not isinstance(x, np.ndarray):
			print("Invalid input: arguments of ndarray type required")	
			return None

		if not x.ndim == 2:
			print("Invalid input: wrong shape of x", x.shape)
			return None

		if self.thetas.ndim == 1 and self.thetas.size == x.shape[1] + 1:
			self.thetas = self.thetas.reshape(x.shape[1] + 1, 1)
		elif not (self.thetas.ndim == 2 and self.thetas.shape == (x.shape[1] + 1, 1)):
			print(f"p Invalid input: wrong shape of {self.thetas}", self.thetas.shape)
			return None
		
		X = np.hstack((np.ones((x.shape[0], 1)), x))
		y_hat = X.dot(self.thetas)
		return np.array(y_hat)

	def loss_elem_(self, y, y_hat):
		"""
		Description: Calculates all the elements (y_pred - y)^2 of the loss function.
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

	def loss_(self, y, y_hat):
		"""
		Description: Calculates the value of loss function.
		"""
		for v in [y, y_hat]:
			if not isinstance(v, np.ndarray):
				print(f"Invalid input: argument {v} of ndarray type required")	
				return None

		for v in [y, y_hat]:
			if not (v.ndim == 2 and v.shape in [(v.size, 1), (1, v.size)]):
				print(f"Invalid input: wrong shape of {v}", v.shape)
				return None

		J_elem = self.loss_elem_(y, y_hat)
		return float(np.sum(J_elem)) / (2 * len(y))


def ex1():

	X = np.array([[1., 1., 2., 3.], [5., 8., 13., 21.], [34., 55., 89., 144.]])
	Y = np.array([[23.], [48.], [218.]])
	mylr = MyLinearRegression(np.array([[1.], [1.], [1.], [1.], [1]]))
	#print("mylr.thetas:", mylr.thetas)

	# Example 0:
	y_hat = mylr.predict_(X) # Output: array([[8.], [48.], [323.]])
	print(y_hat)
	
	# Example 1:
	print(mylr.loss_elem_(Y, y_hat)) # Output: array([[225.], [0.], [11025.]])
	
	# Example 2:
	print(mylr.loss_(Y, y_hat)) # Output: 1875.0
	
	# Example 3:
	mylr.alpha = 1.6e-4
	mylr.max_iter = 200000
	mylr.fit_(X, Y)
	print(mylr.thetas) # Output: array([[18.188..], [2.767..], [-0.374..], [1.392..], [0.017..]])
	
	# Example 4:
	y_hat = mylr.predict_(X) # Output: array([[23.417..], [47.489..], [218.065...]])
	print(y_hat)
	
	# Example 5:
	print(mylr.loss_elem_(Y, y_hat)) # Output: array([[0.174..], [0.260..], [0.004..]])
	
	# Example 6:
	print(mylr.loss_(Y, y_hat)) # Output: 0.0732..

def ex2():
	x = np.array([[0.2, 2., 20.], [0.4, 4., 40.], [0.6, 6., 60.], [0.8, 8., 80.]])
	y = np.array([[19.6], [-2.8], [-25.2], [-47.6]])
	theta = np.array([[42.], [1.], [1.], [1.]])
	mylr = MyLinearRegression(theta)
	
	mylr.alpha = 0.0005 
	mylr.max_iter = 42000 
	mylr.fit_(x, y)
	# Example 0:
	print(mylr.thetas) # Output: array([[41.99..],[0.97..], [0.77..], [-1.20..]])

if __name__ == "__main__":
	ex1()
