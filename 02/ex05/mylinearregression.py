import numpy as np
from gradient import gradient

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
			grad = gradient(x, y ,new_theta)
			# Update new_theta
			for i in range(new_theta.size):
				new_theta[i] -= self.alpha * grad[i]

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
			print(f"Invalid input: wrong shape of {self.thetas}", self.thetas.shape)
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

if __name__ == "__main__":
	ex1()
