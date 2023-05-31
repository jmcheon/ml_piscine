import numpy as np

class MyRidge():
	"""
	Description: My personnal ridge regression class to fit like a boss.
	• __init__, special method, similar to the one you wrote in MyLinearRegression(module06),
	• get_params_, which get the parameters of the estimator,
	• set_params_, which set the parameters of the estimator,
	• loss_, which return the loss between 2 vectors (numpy arrays),
	• loss_elem_, which return a vector corresponding to the squared diffrence between 2 vectors (numpy arrays),
	• predict_, which generates predictions using a linear model,
	• gradient_, which calculates the vectorized regularized gradient,
	• fit_, which fits Ridge regression model to a training dataset.
	"""
	def __init__(self, thetas, alpha=0.001, max_iter=1000, lambda_=0.5):
		if self.check_validation(thetas, alpha, max_iter, lambda_):
			self.alpha = alpha
			self.max_iter = max_iter
			self.lambda_ = lambda_
			if thetas.ndim == 1:
				self.thetas = thetas.reshape(thetas.size, 1)
			else:
				self.thetas = thetas

	def get_params_(self):
		return self.__dict__

	def set_params_(self, thetas=None, alpha=None, max_iter=None, lambda_=None):
		if thetas is not None:
			if not isinstance(thetas, np.ndarray):
				print("Invalid value for thetas. Expected a ndarray.")
				return None
			if thetas.shape != self.thetas.shape:
				print(f"Invalid shape for thetas. Expected shape: {self.thetas.shape}.")
				return None
			self.thetas = thetas
		
		if alpha is not None:
			if not isinstance(alpha, float) or alpha <= 0:
				print("Invalid value for alpha. Expected a positive float.")
				return None
			self.alpha = alpha
		
		if max_iter is not None:
			if not isinstance(max_iter, int) or max_iter <= 0:
				print("Invalid value for max_iter. Expected a positive integer.")
				return None
			self.max_iter = max_iter
		
		if lambda_ is not None:
			if not isinstance(lambda_, float) or lambda_ < 0:
				print("Invalid value for lambda_. Expected a non-negative float.")
				return None
			self.lambda_ = lambda_

	@staticmethod
	def check_validation(thetas, alpha, max_iter, lambda_):
		if not isinstance(thetas, np.ndarray):
			print(f"Invalid input: argument theta of ndarray type required")	
			return False

		if not isinstance(alpha, float) or alpha <= 0:
			print(f"Invalid input: argument alpha of positive float type required")	
			return False

		if not isinstance(max_iter, int) or max_iter <= 0:
			print(f"Invalid input: argument max_iter of positive integer type required")	
			return False 

		if not isinstance(lambda_, float) or lambda_ < 0:
			print(f"Invalid input: argument lambda_ of non-negative float type required")
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
		return mse 
	
	@staticmethod
	def gradient(x, y, theta, lambda_):
		"""
		Computes the regularized linear gradient of three non-empty numpy.ndarray,
		"""
		for v in [x, y, theta]:
			if not isinstance(v, np.ndarray):
				print(f"Invalid input: argument {v} of ndarray type required")  
				return None
	
		if y.ndim == 1:
			y = y.reshape(-1, 1)
		elif not (y.ndim == 2 and y.shape[1] == 1):
			print(f"Invalid input: wrong shape of y", y.shape)  
			return None
	
		if not (x.ndim == 2):
			print(f"Invalid input: wrong shape of x", x.shape)  
			return None
	
		if y.shape[0] != x.shape[0]:
			print(f"Invalid input: unmatched shapes of y and x", y.shape, x.shape)  
			return None
		
		if theta.ndim == 1 and theta.shape[0] == x.shape[1] + 1:
			theta = theta.reshape(x.shape[1] + 1, 1)
		elif not (theta.ndim == 2 and theta.shape[0] == x.shape[1] + 1):
			print(f"Invalid input: wrong shape of theta", theta.shape)  
			return None
	
		if not isinstance(lambda_, float):
			print(f"Invalid input: argument lambda_ of float type required")  
			return None
	
		X = np.hstack((np.ones((x.shape[0], 1)), x))
		theta_ = lambda_ * theta
		theta_[0] = 0
		gradient = np.dot(X.T, (X.dot(theta) - y)) + theta_
		return gradient / x.shape[0]
	
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
		
		if x.shape[0] != y.shape[0]:
			print(f"Invalid input: x, y matrices should be compatible.")
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
			grad = self.gradient(x, y ,new_theta, self.lambda_)
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

		return (y_hat - y + self.lambda_ * (float(np.sum(self.thetas[1:].T.dot(self.thetas[1:]))))) ** 2

	def loss_(self, y, y_hat):
		"""
		Computes the regularized loss of a linear regression model from two non-empty numpy.array, without any for loop.
		"""
		for v in [y, y_hat]:
			if not isinstance(v, np.ndarray):
				print(f"Invalid input: argument {v} of ndarray type required")  
				return None
	
		if y.ndim == 1:
			y = y.reshape(-1, 1)
		elif not (y.ndim == 2 and y.shape[1] == 1):
			print(f"Invalid input: wrong shape of y", y.shape)  
			return None
	
		if y_hat.ndim == 1:
			y_hat = y_hat.reshape(-1, 1)
		elif not (y_hat.ndim == 2 and y_hat.shape[1] == 1):
			print(f"Invalid input: wrong shape of y_hat", y_hat.shape)  
			return None
	
		if y.shape[0] != y_hat.shape[0]:
			print(f"Invalid input: unmatched shapes of y and y_hat", y.shape, y_hat.shape)  
			return None
		
		loss = np.dot((y_hat - y).T, (y_hat - y)) + self.lambda_ * (float(np.sum(self.thetas[1:].T.dot(self.thetas[1:]))))
		return float(loss / (y.shape[0] * 2))

def ex1():

	X = np.array([[1., 1., 2., 3.], [5., 8., 13., 21.], [34., 55., 89., 144.]])
	Y = np.array([[23.], [48.], [218.]])
	mylr = MyRidge(np.array([[1.], [1.], [1.], [1.], [1]]), lambda_=0.0)
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
	mylr = MyRidge(theta, lambda_=0.0)
	
	mylr.alpha = 0.0005 
	mylr.max_iter = 42000 
	mylr.fit_(x, y)
	# Example 0:
	print(mylr.thetas) # Output: array([[41.99..],[0.97..], [0.77..], [-1.20..]])
	mylr.set_params_(lambda_=2.0)
	print(mylr.get_params_())

if __name__ == "__main__":
	ex2()
