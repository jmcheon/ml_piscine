import numpy as np

class MyLogisticRegression():
	"""
	Description: My personnal logistic regression to classify things.
	Methodes: fit_(self, x, y), predict_(self, x), loss_elem_(self, y, y_hat), loss_(self, y, y_hat).
	"""
	# We consider l2 penality only. One may wants to implement other penalities
	supported_penalties = ['l2'] 
	def __init__(self, thetas, alpha=0.001, max_iter=1000, penalty='l2', lambda_=1.0):
		if self.check_validation(thetas, alpha, max_iter, penalty, lambda_):
			self.alpha = alpha
			self.max_iter = max_iter
			self.penalty = penalty
			self.lambda_ = lambda_ if penalty in self.supported_penalties else 0.0
			if thetas.ndim == 1:
				self.thetas = thetas.reshape(thetas.size, 1)
			else:
				self.thetas = thetas

	@staticmethod
	def check_validation(thetas, alpha, max_iter, penalty, lambda_):
		if not isinstance(thetas, np.ndarray):
			print(f"Invalid input: argument theta of ndarray type required")
			return False

		if not isinstance(alpha, float):
			print(f"Invalid input: argument alpha of float type required")
			return False

		if not isinstance(max_iter, int):
			print(f"Invalid input: argument max_iter of int type required")
			return False 

		if not penalty in ['l2', None]:
			print(f"Invalid input: invalid value of penalty {penalty}")
			return False

		if not isinstance(lambda_, float):
			print(f"Invalid input: argument lambda_ of float type required")
			return False

		return True

	@staticmethod
	def gradient(x, y, theta, lambda_=0.0):
		"""
		Computes a gradient vector from three non-empty numpy.ndarray, without any for-loop. The three arrays must have compatible
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
	
		if x.shape[0] != y.shape[0]:
			print(f"Invalid input: two vectors of compatible shape are required")
			return None
	
		if theta.ndim == 1 and theta.size == x.shape[1] + 1:
			theta = theta.reshape(x.shape[1] + 1, 1)
		elif not (theta.ndim == 2 and theta.shape == (x.shape[1] + 1, 1)):
			print(f"Invalid input: wrong shape of theta", theta.shape)
			return None

		if not isinstance(lambda_, float):
			print(f"Invalid input: argument lambda_ of float type required")  
			return None
	
		X = np.hstack((np.ones((x.shape[0], 1)), x))
		y_hat = np.array(1 / (1 + np.exp(-X.dot(theta))))
		theta_ = lambda_ * theta
		theta_[0] = 0
		gradient = np.dot(X.T, (y_hat - y)) + theta_
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

		if self.thetas.ndim == 1 and self.thetas.size == x.shape[1] + 1:
			self.thetas = self.thetas.reshape(x.shape[1] + 1, 1)
		elif not (self.thetas.ndim == 2 and self.thetas.shape == (x.shape[1] + 1, 1)):
			print(f"Invalid input: wrong shape of {self.thetas}", self.thetas.shape)
			return None

		new_theta = np.copy(self.thetas.astype("float64"))
		for _ in range(self.max_iter):
			# Compute gradient descent
			if self.penalty == None:
				grad = self.gradient(x, y ,new_theta)
			elif self.penalty == 'l2':
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
		Computes the vector of prediction y_hat from two non-empty numpy.ndarray.
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
			print(f"Invalid input: wrong shape of {self.thetas}", self.thetas.shape)
			return None
	
		X = np.hstack((np.ones((x.shape[0], 1)), x))
		return np.array(1 / (1 + np.exp(-X.dot(self.thetas))))

	def loss_elem_(self, y, y_hat, eps=1e-15):
		"""
		Description: Calculates all the elements of the logistic loss function.
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

		y_hat = np.clip(y_hat, eps, 1 - eps)  
		loss_elem = -(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))
		return loss_elem

	def loss_(self, y, y_hat, eps=1e-15):
		"""
		Compute the logistic loss value.
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
	X = np.array([[1., 1., 2., 3.], [5., 8., 13., 21.], [3., 5., 9., 14.]])
	Y = np.array([[1], [0], [1]])
	thetas = np.array([[2], [0.5], [7.1], [-4.3], [2.09]])
	mylr = MyLogisticRegression(thetas, lambda_=0.0)
	# Example 0:
	y_hat = mylr.predict_(X)
	print(y_hat) # Output: array([[0.99930437], [1. ], [1. ]])

	# Example 1:
	#print(np.mean(mylr.loss_elem_(Y, y_hat)))
	print(mylr.loss_(Y, y_hat)) # Output: 11.513157421577004

	# Example 2:
	mylr.fit_(X, Y)
	print(mylr.thetas)
	# Output: array([[ 2.11826435] [ 0.10154334] [ 6.43942899] [-5.10817488] [ 0.6212541 ]])
	# Example 3:
	y_hat = mylr.predict_(X)
	print(y_hat) # Output: array([[0.57606717] [0.68599807] [0.06562156]])

	# Example 4:
	#print(np.mean(mylr.loss_elem_(Y, y_hat)))
	print(mylr.loss_(Y, y_hat)) # Output: 1.4779126923052268

def ex2():
	thetas = np.array([[-2.4], [-1.5], [0.3], [-1.4], [0.7]])

	# Example 1:
	model1 = MyLogisticRegression(thetas, lambda_=5.0)
	print(model1.penalty) # Output ’l2’
	print(model1.lambda_) # Output 5.0

	# Example 2:
	model2 = MyLogisticRegression(thetas, penalty=None)
	print(model2.penalty) # Output None
	print(model2.lambda_) # Output 0.0

	# Example 3:
	model3 = MyLogisticRegression(thetas, penalty=None, lambda_=2.0)
	print(model3.penalty) # Output None
	print(model3.lambda_) # Output 0.0

def ex3():
	x = np.array([[0, 2, 3, 4],
	[2, 4, 5, 5],
	[1, 3, 2, 7]])
	y = np.array([[0], [1], [1]])
	thetas = np.array([[-2.4], [-1.5], [0.3], [-1.4], [0.7]])

	model1 = MyLogisticRegression(thetas, lambda_=0.0)

	# Example 1.2:
	print(model1.gradient(x, y, thetas, 1.0)) # Output: array([[-0.55711039], [-1.40334809], [-1.91756886], [-2.56737958], [-3.03924017]])
	# Example 2.2:
	print(model1.gradient(x, y, thetas, 0.5)) # Output: array([[-0.55711039], [-1.15334809], [-1.96756886], [-2.33404624], [-3.15590684]])
	# Example 3.2:
	print(model1.gradient(x, y, thetas, 0.0)) # Output: array([[-0.55711039], [-0.90334809], [-2.01756886], [-2.10071291], [-3.27257351]])


if __name__ == "__main__":
	ex2()
