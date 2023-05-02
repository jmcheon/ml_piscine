

class Matrix:

	def __init__(self, data):
		"""
		data: list of lists
		shape: the dimensions of the matrix as a tuple (rows, columns)
		"""
		self.data = []
		# the elements of the matrix as a list of lists: Matrix([[1.0, 2.0], [3.0, 4.0]])
		if isinstance(data, list):
			if all(isinstance(elem, list) and len(data[0]) == len(elem) and all(isinstance(i, float) for i in elem) for elem in data):
				self.data = data
				self.shape = (len(data), len(data[0])) 
		# a shape: Matrix((3, 3)) (the matrix will be filled with zeros by default)
		elif isinstance(data, tuple) and len(data) == 2 and all(isinstance(elem, int) and elem >= 0 for elem in data):
			for i in range(data[0]):
				row = []
				for j in range(data[1]):
					row.append(0)
				self.data.append(row)
				self.shape = (data[0], data[1])
		else:
			raise ValueError("Invalid form of data,", data)

	def T(self):
		".T() method which returns the transpose of the matrix."
		transposed = []
		for j in range(self.shape[1]):
			row = []
			for i in range(self.shape[0]):
				row.append(self.data[i][j])
			transposed.append(row)
		return Matrix(transposed)

	# add : only matrices of same dimensions.
	def __add__(self, other):
		if not isinstance(other, Matrix):
			raise TypeError("unsupported operand type(s) for +: '{}' and '{}'".format(type(self), type(other)))
			#raise TypeError(f"Invalid input: {func.__name__} requires a Matrix object.")
		if self.shape != other.shape:
			raise ValueError(f"Invalid input: addition requires a Matrix of same shape.")
		result = [[self.data[i][j] + other.data[i][j] for j in range(self.shape[1])] for i in range(self.shape[0])]
		return Matrix(result)

	def __radd__(self, other):
		if not isinstance(other, Matrix):
			raise TypeError("unsupported operand type(s) for +: '{}' and '{}'".format(type(self), type(other)))
		if self.shape != other.shape:
			raise ValueError("Invalid input: __add__ requires matrics of the same shape.")
		return other + self

	# sub : only matrices of same dimensions.
	def __sub__(self, other):
		if not isinstance(other, Matrix):
			raise TypeError("unsupported operand type(s) for +: '{}' and '{}'".format(type(self), type(other)))
			#raise TypeError(f"Invalid input: {func.__name__} requires a Matrix object.")
		if self.shape != other.shape:
			raise ValueError(f"Invalid input: subtraction requires a Matrix of same shape.")
		result = [[self.data[i][j] - other.data[i][j] for j in range(self.shape[1])] for i in range(self.shape[0])]
		return Matrix(result)

	def __rsub__(self, other):
		if not isinstance(other, Matrix):
			raise TypeError("unsupported operand type(s) for +: '{}' and '{}'".format(type(self), type(other)))
		if self.shape != other.shape:
			raise ValueError("Invalid input: __add__ requires matrics of the same shape.")
		return other - self

	# div : only scalars.
	def __truediv__(self, scalar):
		if isinstance(scalar, Matrix):
			raise NotImplementedError("Division with a Matrix object is not implemented.")
		if not any(isinstance(scalar, scalar_type) for scalar_type in [int, float, complex]):
			raise TypeError("Invalid input of scalar value.")
		if scalar == 0:
			raise ValueError("Can't divide by 0.")
		result = [[self.data[i][j] / scalar for j in range(self.shape[1])] for i in range(self.shape[0])]
		return Matrix(result)

	def __rtruediv__(self, scalar):
		raise NotImplementedError("Division of a scalar by a Matrix object is not defined here.")

	# mul : scalars, vectors and matrices , can have errors with vectors and matrices,
	# returns a Vector if we perform Matrix * Vector mutliplication.
	def __mul__(self, other):
		if any(isinstance(other, scalar_type) for scalar_type in [int, float, complex]):
			result = [[self.data[i][j] * other for j in range(self.shape[1])] for i in range(self.shape[0])]
			return Matrix(result)
		elif isinstance(other, Vector):
			if self.shape[1] != other.shape[0]:
				raise ValueError("Matrices cannot be multiplied, dimensions don't match.")
			result = [[sum([self.data[i][k] * other.data[k][j] for k in range(self.shape[1])]) for j in range(other.shape[1])] for i in range(self.shape[0])]
			return Vector(result)
		elif isinstance(other, Matrix):
			if self.shape[1] != other.shape[0]:
				raise ValueError("Matrices cannot be multiplied, dimensions don't match.")
			result = [[sum([self.data[i][k] * other.data[k][j] for k in range(self.shape[1])]) for j in range(other.shape[1])] for i in range(self.shape[0])]
			return Matrix(result)
		else:
			raise TypeError("Invalid type of input value.")

	def	__rmul__(self, x):
		return self * x

	def __str__(self):
		txt = f"Matrix({self.data}) {self.shape}"
		return txt

	def __repr__(self):
		txt = f"Matrix({self.data}) {self.shape}"
		return txt

class Vector(Matrix):
	
	def __init__(self, data):
		self.data = []
		# when data is a list
		if isinstance(data, list):
			# initialize a list of a list of floats : Vector([[0.0, 1.0, 2.0, 3.0]])
			if len(data) == 1 and isinstance(data[0], list) and len(data[0]) > 0 and all(type(i) in [int, float] for i in data[0]):	
				self.data = data
				self.shape = (1, len(data[0]))
			# initialize a list of lists of single float : Vector([[0.0], [1.0], [2.0], [3.0]])
			elif all(isinstance(elem, list) and len(elem) == 1 and all(type(i) in [int, float] for i in elem) for elem in data):
				self.data = data
				self.shape = (len(data), 1)
			else:
				raise ValueError("Invalid form of list,", data)
		else:
			raise ValueError("Invalid form of data,", data)

	def dot(self, other):
		if not isinstance(other, Vector):
			raise TypeError("unsupported operand type(s) for +: '{}' and '{}'".format(type(self), type(other)))
		if self.shape[1] != other.shape[0]:
			raise TypeError("Invalid input: dot product requires a Vector of compatible shape.")
		result = 0.0
		for i in range(self.shape[0]):
			for j in range(self.shape[1]):
				result += self.data[i][j] * other.data[j][i]
		return result

	def T(self):
		".T() method which returns the transpose of the matrix."
		transposed = []
		for j in range(self.shape[1]):
			row = []
			for i in range(self.shape[0]):
				row.append(self.data[i][j])
			transposed.append(row)
		return Matrix(transposed)

	def __add__(self, other):
		if not isinstance(other, Vector):
			raise TypeError("unsupported operand type(s) for +: '{}' and '{}'".format(type(self), type(other)))
		if self.shape != other.shape:
			raise ValueError("Invalid input: __add__ requires vectors of the same shape.")
		result = [[self.data[i][j] + other.data[i][j] for j in range(self.shape[1])] for i in range(self.shape[0])]
		return Vector(result)
	
	def __sub__(self, other):
		if not isinstance(other, Vector):
			raise TypeError("unsupported operand type(s) for -: '{}' and '{}'".format(type(self), type(other)))
		if self.shape != other.shape:
			raise ValueError("Invalid input: __sub__ requires vectors of the same shape.")
		result = [[self.data[i][j] - other.data[i][j] for j in range(self.shape[1])] for i in range(self.shape[0])]
		return Vector(result)
	
	def __mul__(self, other):
		if any(isinstance(other, scalar_type) for scalar_type in [int, float, complex]):
			result = [[self.data[i][j] * other for j in range(self.shape[1])] for i in range(self.shape[0])]
			return Vector(result)
		elif isinstance(other, Vector):
			if self.shape[1] != other.shape[0]:
				raise ValueError("Vectors cannot be multiplied, dimensions don't match.")
			result = [[self.data[i][j] * other for j in range(self.shape[1])] for i in range(self.shape[0])]
			return Vector(result)
		elif isinstance(other, Matrix):
			if self.shape[1] != other.shape[0]:
				raise ValueError("Matrices cannot be multiplied, dimensions don't match.")
			result = [[sum([self.data[i][k] * other.data[k][j] for k in range(self.shape[1])]) for j in range(other.shape[1])] for i in range(self.shape[0])]
			return Matrix(result)
		else:
			raise TypeError("Invalid type of input value.")
	
	def __truediv__(self, scalar):
		if isinstance(scalar, Vector):
			raise NotImplementedError("Vector division is not implemented.")
		elif not any(isinstance(scalar, scalar_type) for scalar_type in [int, float, complex]):
			raise TypeError("Invalid input of scalar value.")
		if scalar == 0:
			raise ValueError("Can't divide by 0.")
		result = [[self.data[i][j] / scalar for j in range(self.shape[1])] for i in range(self.shape[0])]
		return Vector(result)

	def __str__(self):
		txt = f"Vector({self.data}) {self.shape}"
		return txt

	def __repr__(self):
		txt = f"Vector({self.data}) {self.shape}"
		return txt

def ex1():
	m1 = Matrix([[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]])
	print(m1)
	print(m1.shape) # Output: (3, 2)
	print(m1.T()) # Output: Matrix([[0., 2., 4.], [1., 3., 5.]])
	print(m1.T().shape) # Output: (2, 3)

	m1 = Matrix([[0., 2., 4.], [1., 3., 5.]])
	print(m1)
	print(m1.shape) # Output: (2, 3)
	print(m1.T()) # Output: Matrix([[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]])
	print(m1.T().shape) # Output: (3, 2)

def ex_matrix_by_matrix():
	m1 = Matrix([[0.0, 1.0, 2.0, 3.0], 
				[0.0, 2.0, 4.0, 6.0]]) # (2, 4)

	m2 = Matrix([[0.0, 1.0, 2.0, 3.0], 
				[4.0, 5.0, 6.0, 7.0]]) # (2, 4)

	m3 = Matrix([[0.0, 1.0], 
				[2.0, 3.0], 
				[4.0, 5.0], 
				[6.0, 7.0]]) # (4, 2)

	print(m1 + m2) # Output: Matrix([0.0, 2.0, 4.0, 6.0], [4.0, 7.0, 10.0, 13.0])
	print(m2 + m1) # Output: Matrix([0.0, 2.0, 4.0, 6.0], [4.0, 7.0, 10.0, 13.0])

	print(m1 - m2) # Output: Matrix([0.0, 0.0, 0.0, 0.0], [-4.0, -3.0, -2.0, -1.0])
	print(m2 - m1) # Output: Matrix([0.0, 0.0, 0.0, 0.0], [4.0, 3.0, 2.0, 1.0])

	print(m1 / 2) # Output: Matrix([0.0, 2.0, 4.0, 6.0], [4.0, 7.0, 10.0, 13.0])

	print(m3 * m1) # Output: Matrix([[0.0, 2.0, 4.0, 6.0], [0.0, 8.0, 16.0, 24.0], [0.0, 14.0, 28.0, 42.0], [0.0, 13.0, 40.0, 60.0]]) (4, 4)
	print(m1 * m3) # Output: Matrix([[28.0, 34.0], [56.0, 68.0]]) (2, 2)

def ex_matrix_by_vector():
	m1 = Matrix([[0.0, 1.0, 2.0], 
				[0.0, 2.0, 4.0]]) # (2, 3)

	m2 = Matrix([[0.0, 1.0, 2.0]]) # (1, 3)

	v1 = Vector([[1], [2], [3]]) # (3, 1)

	print(m1 * v1) # Output: Matrix([[8], [16]]) # Or: Vector([[8], [16])
	print(m1 * 3) # 
	print(2 * m1) # 
	print(v1 * m2) # Output: Matrix([[0.0, 1.0, 2.0], [0.0, 2.0, 4.0], [0.0, 3.0, 6.0]]) (3, 3)

def ex_vector_by_vector():
	v1 = Vector([[1], [2], [3]]) # (3, 1)
	v2 = Vector([[2], [4], [8]])
	print(v1 + v2) # Output: Vector([[3],[6],[11]])
	print(v2 + v1) # Output: Vector([[3],[6],[11]])
	#print(v1 + 1) # Output: error
	#print(v2 + "hi") # Output: error 
	#print(v2 + "hi") # Output: error 
	print(v1 - v2) # Output: Vector([[-1],[-2],[-5]])
	print(v2 - v1) # Output: Vector([[1],[2],[5]])

if __name__ == "__main__":
	#ex_matrix_by_matrix()
	#ex_matrix_by_vector()
	ex_vector_by_vector()
