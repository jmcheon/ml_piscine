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
	def check_validation(theta, alpha, max_iter):
		if not isinstance(theta, np.ndarray):
			print(f"Invalid input: argument theta of ndarray type required")	
			return False
		if theta.ndim == 1:
			pass
		elif not (theta.ndim == 2 and theta.shape == (2, 1)):
			print(f"Invalid input: wrong shape of {theta}", theta.shape)
			return False

		if not isinstance(alpha, float):
			print(f"Invalid input: argument alpha of float type required")	
			return False

		if not isinstance(max_iter, int):
			print(f"Invalid input: argument max_iter of int type required")	
			return False 
		return True
	
	def fit_(self, x, y):
		"""
		Description: Fits the model to the training dataset contained in x and y.
		"""
		# arguments type varification
		for v in [x, y]:
			if not isinstance(v, np.ndarray):
				print(f"Invalid input: argument {v} of ndarray type required")	
				return None

		# vector arguments shape varification
		v = [x, y]
		for i in range(len(v)): 
			if v[i].ndim == 1:
				v[i] = v[i].reshape(v[i].size, 1)
			elif not (v[i].ndim == 2 and v[i].shape[1] == 1):
				print(f"Invalid input: wrong shape of {v[i]}", v[i].shape)
				return None
		x, y = v

		# Weights to update: alpha * mean((y_hat - y) * x) 
		# Bias to update: alpha * mean(y_hat - y)
		X = np.hstack((np.ones((x.shape[0], 1)), x))
		new_theta = np.copy(self.thetas.astype("float64"))
		for _ in range(self.max_iter):
			y_hat = X.dot(new_theta)
			# Compute gradient descent
			#gradient = simple_gradient(x, y ,new_theta)
			b_gradient = (y_hat - y).mean()
			w_gradient = ((y_hat - y) * x).mean()

			# Update new_theta
			new_theta[0] -= self.alpha * b_gradient
			new_theta[1] -= self.alpha * w_gradient

		self.thetas = new_theta
		return self.thetas

	def predict_(self, x):
		"""
		Description: Computes the vector of prediction y_hat from two non-empty numpy.array.
		"""
		if not isinstance(x, np.ndarray):
			print("Invalid input: arguments of ndarray type required")	
			return None

		if x.ndim == 1:
			x = x.reshape(x.size, 1)
		elif not (x.ndim == 2 and x.shape[1] == 1):
			print("Invalid input: wrong shape of x", x.shape)
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
	x = np.array([[12.4956442], [21.5007972], [31.5527382], [48.9145838], [57.5088733]])
	y = np.array([[37.4013816], [36.1473236], [45.7655287], [46.6793434], [59.5585554]])
	lr1 = MyLinearRegression(np.array([[2], [0.7]]))
	# Example 0.0:
	y_hat = lr1.predict_(x) # Output: array([[10.74695094], [17.05055804], [24.08691674], [36.24020866], [42.25621131]])
	print(y_hat)
	
	# Example 0.1:
	print(lr1.loss_elem_(y, y_hat)) # Output: array([[710.45867381], [364.68645485], [469.96221651], [108.97553412], [299.37111101]])
	
	# Example 0.2:
	print(lr1.loss_(y, y_hat)) # Output: 195.34539903032385
	
	# Example 1.0:
	lr2 = MyLinearRegression(np.array([[1], [1]]), 5e-8, 1500000)
	lr2.fit_(x, y)
	print(lr2.thetas) # Output: array([[1.40709365], [1.1150909 ]])
	
	# Example 1.1:
	y_hat = lr2.predict_(x) # Output: array([[15.3408728 ], [25.38243697], [36.59126492], [55.95130097], [65.53471499]])
	print(y_hat)
	
	# Example 1.2:
	print(lr2.loss_elem_(y, y_hat)) # Output: array([[486.66604863], [115.88278416], [ 84.16711596], [ 85.96919719], [ 35.71448348]])
	
	# Example 1.3:
	print(lr2.loss_(y, y_hat)) # Output: 80.83996294128525

if __name__ == "__main__":
	ex1()
