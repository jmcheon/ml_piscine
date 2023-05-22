import numpy as np

def add_polynomial_features(x, power):
    """
    Add polynomial features to vector x by raising its values up to the power given in the argument.

    Args:
    x: has to be a numpy.array, a vector or matrix of dimensions m * n.
    power: has to be an int, the power up to which the components of vector x are going to be raised.

    Return:
    The matrix of polynomial features as a numpy.array, of dimension m * p,
    containing the polynomial feature values for all training examples, where p = (n + 1) * power.

    None if x is an empty numpy.array.
    None if x or power is not of the expected type.

    Raises:
    This function should not raise any Exception.
    """
    if not isinstance(power, int):
        print(f"Invalid input: argument power of int type required")  
        return None 
    if not isinstance(x, np.ndarray):
        print(f"Invalid input: argument x of ndarray type required")  
        return None

    if x.ndim == 1:
        x = x.reshape(-1, 1)

    if power < 0:
        return None
    elif power == 0:
        return np.ones((x.shape[0], 1))
    elif power == 1:
        return np.copy(x)
    else:
        x_poly = np.ones((x.shape[0], x.shape[1] * power))
        for i in range(power):
            x_poly[:, i * x.shape[1] : (i + 1) * x.shape[1]] = x ** (i + 1)
        return x_poly
	
def ex1():
	x = np.arange(1,6).reshape(-1, 1)
	print("x:", x, x.shape)

	# Example 0:
	print(add_polynomial_features(x, 3))
	# Example 1:
	print(add_polynomial_features(x, 6))

if __name__ == "__main__":
	ex1()
