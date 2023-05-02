from matrix import Vector

if __name__ == "__main__":

	# when data is a list
	print("\n << when data is a list >>")
	print("\n << Column vector of shape n * 1 >>")
	print(Vector([[0.0], [1.0], [2.0], [3.0]]).data)
	print(Vector([[0.0], [-1.0], [2.0], [3.0]]).data)
#	Vector([[0], [1.0], [2.0], [3.0]])
#	Vector([[0.0, 1.0], [1.0], [2.0], [3.0]])
#	Vector([[], [1.0], [2.0], [3.0]])

	print("\n << Row vector of shape 1 * n >>")
	print(Vector([[0.0, 1.0, 2.0, 3.0]]).data)
	print(Vector([[0.0, -1.0, 2.0, 3.0]]).data)
	print(Vector([[-0.0, 1.0, 2.0, 3.0]]).data)
	#Vector([[0, 1.0, 2.0, 3.0]])
	#Vector([[[], 1.0, 2.0, 3.0]])
	#Vector([1.0, [0.0, 1.0, 2.0, 3.0]])
	#Vector([[]])

	# dot product
	print("\n << dot product >>")
	v1 = Vector([[2.0, 3.0, 1.0]])
	v2 = Vector([[2.0], [3.0], [1.0]])
	#v1 = Vector([[0.0], [1.0], [2.0], [3.0]])
	#v2 = Vector([[2.0], [1.5], [2.25], [4.0]])
	print(v1)
	print(v1.dot(v2))

	# transpose
	print("\n << transpose >>")
	print(v1.T())
	print(v1.T().shape)
	print(v2.T())
	print(v2.T().shape)

	# division by scalar
	v3 = v1 / 2.0
	print(v3.data)
	#v3 = Vector(4) / 0
	#v3 = Vector(4) / None
	#v3 = None / Vector(4) 
	#v3 = 3 / Vector(4) 

	# addition by scalar
	v = Vector([[0.0], [1.0], [2.0], [3.0]])
	v2 = Vector([[1.0], [1.0], [1.0], [1.0]])
	print((v + v2).data)
	print((v - v2).data != (v2 - v).data)

	# multiplication by scalar
	print(v * 4) 
	print(4.0 * v) 
	#v * "hello"
