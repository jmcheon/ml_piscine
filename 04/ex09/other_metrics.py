import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def accuracy_score_(y, y_hat):
	"""
	Compute the accuracy score.

	Args:
	y:a numpy.ndarray for the correct labels
	y_hat:a numpy.ndarray for the predicted labels

	Returns:
	The accuracy score as a float.
	None on any error.

	Raises:
	This function should not raise any Exception.
	"""
	for v in [y, y_hat]:
		if not isinstance(v, np.ndarray):
			print(f"Invalid input: argument {v} of ndarray type required")
			return None

	if y.shape != y_hat.shape:
		print("Invalid input: y and y_hat must have the same shape.")
		return None

    # Check if y and y_hat are numeric arrays
	if np.issubdtype(y.dtype, np.number) and np.issubdtype(y_hat.dtype, np.number):
        # Check if y and y_hat are binary arrays
		if not (np.array_equal(np.unique(y), [0, 1]) and np.all(np.logical_or(y_hat == 0, y_hat == 1))):
			print("Invalid input: y and y_hat must contain binary values.")
			return None
    # Check if y and y_hat are string arrays
	elif not np.issubdtype(y.dtype, str) and np.issubdtype(y_hat.dtype, str):
        # Continue with your code for calculating the precision score for string arrays
		print("Invalid input: y and y_hat should be numeric arrays or string arrays.")

	correct_predictions = np.sum(y == y_hat)
	accuracy = correct_predictions / len(y)
	return float(accuracy)

def precision_score_(y, y_hat, pos_label=1):
	"""
	Compute the precision score.

	Args:
	y:a numpy.ndarray for the correct labels
	y_hat:a numpy.ndarray for the predicted labels
	pos_label: str or int, the class on which to report the precision_score (default=1)

	Return:
	The precision score as a float.
	None on any error.

	Raises:
	This function should not raise any Exception.
	"""
	for v in [y, y_hat]:
		if not isinstance(v, np.ndarray):
			print(f"Invalid input: argument {v} of ndarray type required")
			return None

	if not isinstance(pos_label, (str, int)):
		print(f"Invalid input: argument pos_label of str or int type required")
		return None

	if y.shape != y_hat.shape:
		print("Invalid input: y and y_hat must have the same shape.")
		return None

   	# Check if y and y_hat are numeric arrays
	if np.issubdtype(y.dtype, np.number) and np.issubdtype(y_hat.dtype, np.number):
        # Check if y and y_hat are binary arrays
		if not (np.array_equal(np.unique(y), [0, 1]) and np.all(np.logical_or(y_hat == 0, y_hat == 1))):
			print("Invalid input: y and y_hat must contain binary values.")
			return None
    # Check if y and y_hat are string arrays
	elif np.issubdtype(y.dtype, str) and np.issubdtype(y_hat.dtype, str):
		is_valid_pos_label = np.isin(pos_label, y) and np.isin(pos_label, y_hat)
		valid_labels = np.unique(np.concatenate((y, y_hat)))
		if not is_valid_pos_label:
			print(f"Invalid input: pos_label={pos_label} is not a valid label. It should be one of {valid_labels}")
			return None
	else:
		print("Invalid input: y and y_hat should be numeric arrays or string arrays.")
	
	true_positives = np.sum((y == pos_label) & (y_hat == pos_label))
	predicted_positives = np.sum(y_hat == pos_label)
	if predicted_positives == 0:
		print(f"UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 due to no predicted samples.")
		return None
	precision = true_positives / predicted_positives
	return float(precision)

def recall_score_(y, y_hat, pos_label=1):
	"""
	Compute the recall score.

	Args:
	y:a numpy.ndarray for the correct labels
	y_hat:a numpy.ndarray for the predicted labels
	pos_label: str or int, the class on which to report the precision_score (default=1)

	Return:
	The recall score as a float.
	None on any error.

	Raises:
	This function should not raise any Exception.
	"""
	for v in [y, y_hat]:
		if not isinstance(v, np.ndarray):
			print(f"Invalid input: argument {v} of ndarray type required")
			return None

	if not isinstance(pos_label, (str, int)):
		print(f"Invalid input: argument pos_label of str or int type required")
		return None

	if y.shape != y_hat.shape:
		print("Invalid input: y and y_hat must have the same shape.")
		return None

    # Check if y and y_hat are numeric arrays
	if np.issubdtype(y.dtype, np.number) and np.issubdtype(y_hat.dtype, np.number):
        # Check if y and y_hat are binary arrays
		if not (np.array_equal(np.unique(y), [0, 1]) and np.all(np.logical_or(y_hat == 0, y_hat == 1))):
			print("Invalid input: y and y_hat must contain binary values.")
			return None
    # Check if y and y_hat are string arrays
	elif np.issubdtype(y.dtype, str) and np.issubdtype(y_hat.dtype, str):
		is_valid_pos_label = np.isin(pos_label, y) and np.isin(pos_label, y_hat)
		valid_labels = np.unique(np.concatenate((y, y_hat)))
		if not is_valid_pos_label:
			print(f"Invalid input: pos_label={pos_label} is not a valid label. It should be one of {valid_labels}")
			return None
	else:
		print("Invalid input: y and y_hat should be numeric arrays or string arrays.")
	

	true_positives = np.sum((y == pos_label) & (y_hat == pos_label))
	actual_positives = np.sum(y == pos_label)
	recall = true_positives / actual_positives
	return float(recall)

def f1_score_(y, y_hat, pos_label=1):
	"""
	Compute the f1 score.

	Args:
	y:a numpy.ndarray for the correct labels
	y_hat:a numpy.ndarray for the predicted labels
	pos_label: str or int, the class on which to report the precision_score (default=1)

	Returns:
	The f1 score as a float.
	None on any error.

	Raises:
	This function should not raise any Exception.
	"""
	for v in [y, y_hat]:
		if not isinstance(v, np.ndarray):
			print(f"Invalid input: argument {v} of ndarray type required")
			return None

	if not isinstance(pos_label, (str, int)):
		print(f"Invalid input: argument pos_label of str or int type required")
		return None

	if y.shape != y_hat.shape:
		print("Invalid input: y and y_hat must have the same shape.")
		return None

    # Check if y and y_hat are numeric arrays
	if np.issubdtype(y.dtype, np.number) and np.issubdtype(y_hat.dtype, np.number):
        # Check if y and y_hat are binary arrays
		if not (np.array_equal(np.unique(y), [0, 1]) and np.all(np.logical_or(y_hat == 0, y_hat == 1))):
			print("Invalid input: y and y_hat must contain binary values.")
			return None
    # Check if y and y_hat are string arrays
	elif np.issubdtype(y.dtype, str) and np.issubdtype(y_hat.dtype, str):
		is_valid_pos_label = np.isin(pos_label, y) and np.isin(pos_label, y_hat)
		valid_labels = np.unique(np.concatenate((y, y_hat)))
		if not is_valid_pos_label:
			print(f"Invalid input: pos_label={pos_label} is not a valid label. It should be one of {valid_labels}")
			return None
	else:
		print("Invalid input: y and y_hat should be numeric arrays or string arrays.")
	
	precision = precision_score_(y, y_hat, pos_label)
	recall = recall_score_(y, y_hat, pos_label)
	if precision == None or recall == None:
		return None
	f1_score = 2 * (precision * recall) / (precision + recall)
	return float(f1_score)

def ex1():
	print("\nExample 1")
	y_hat = np.array([1, 1, 0, 1, 0, 0, 1, 1]).reshape((-1, 1))
	y = np.array([1, 0, 0, 1, 0, 1, 0, 0]).reshape((-1, 1))

	print("Accuracy")
	## your implementation
	print("mine:", accuracy_score_(y, y_hat)) ## Output: 0.5
	## sklearn implementation
	print("real:", accuracy_score(y, y_hat)) ## Output: 0.5

	print("Precision")
	## your implementation
	print("mine:", precision_score_(y, y_hat)) ## Output: 0.4
	## sklearn implementation
	print("real:", precision_score(y, y_hat)) ## Output: 0.4

	print("Recall")
	## your implementation
	print("mine:", recall_score_(y, y_hat)) ## Output: 0.6666666666666666
	## sklearn implementation
	print("real:", recall_score(y, y_hat)) ## Output: 0.6666666666666666

	print("F1-score")
	## your implementation
	print("mine:", f1_score_(y, y_hat)) ## Output: 0.5
	## sklearn implementation
	print("real:", f1_score(y, y_hat)) ## Output: 0.5
	
def ex2():
	print("\nExample 2")
	y_hat = np.array(['norminet', 'dog', 'norminet', 'norminet', 'dog', 'dog', 'dog', 'dog'])
	y = np.array(['dog', 'dog', 'norminet', 'norminet', 'dog', 'norminet', 'dog', 'norminet'])
	print("Accuracy")
	## your implementation
	print("mine:", accuracy_score_(y, y_hat)) ## Output: 0.625
	## sklearn implementation
	print("real:", accuracy_score(y, y_hat)) ## Output: 0.625

	print("Precision")
	## your implementation
	print("mine:", precision_score_(y, y_hat, pos_label='dog')) ## Output: 0.6
	## sklearn implementation
	print("real:", precision_score(y, y_hat, pos_label='dog')) ## Output: 0.6

	print("Recall")
	## your implementation
	print("mine:", recall_score_(y, y_hat, pos_label='dog')) ## Output: 0.75
	## sklearn implementation
	print("real:", recall_score(y, y_hat, pos_label='dog')) ## Output: 0.75

	print("F1-score")
	## your implementation
	print("mine:", f1_score_(y, y_hat, pos_label='dog')) ## Output: 0.6666666666666665
	## sklearn implementation
	print("real:", f1_score(y, y_hat, pos_label='dog')) ## Output: 0.6666666666666665

def ex3():
	print("\nExample 3")
	# Example 3:
	y_hat = np.array(['norminet', 'dog', 'norminet', 'norminet', 'dog', 'dog', 'dog', 'dog'])
	y = np.array(['dog', 'dog', 'norminet', 'norminet', 'dog', 'norminet', 'dog', 'norminet'])
	print("Precision")
	## your implementation
	print("mine:", precision_score_(y, y_hat, pos_label='norminet')) ## Output: 0.6666666666666666
	## sklearn implementation
	print("real:", precision_score(y, y_hat, pos_label='norminet')) ## Output: 0.6666666666666666
	print("Recall")
	## your implementation
	print("mine:", recall_score_(y, y_hat, pos_label='norminet')) ## Output: 0.5
	## sklearn implementation
	print("real:", recall_score(y, y_hat, pos_label='norminet')) ## Output: 0.5

	print("F1-score")
	## your implementation
	print("mine:", f1_score_(y, y_hat, pos_label='norminet')) ## Output: 0.5714285714285715
	## sklearn implementation
	print("real:", f1_score(y, y_hat, pos_label='norminet')) ## Output: 0.5714285714285715

def binary_continuous_target_test():
	print("\nExample for binary_continuous_target_test")
	y_hat = np.random.rand(8, 1).reshape((-1, 1))
	y = np.array([1, 0, 0, 1, 0, 1, 0, 0]).reshape((-1, 1))
	print("y_hat:", y_hat, y_hat.shape)

	lst_funcs_name = ["Accuracy", "Precision", "Recall", "F1-score"]
	lst_my_funcs = [accuracy_score_, precision_score_, recall_score_, f1_score_]
	lst_sklearn_funcs = [accuracy_score, precision_score, recall_score, f1_score]

	for i in range(len(lst_my_funcs)):
		print(lst_funcs_name[i])
		print("mine:", lst_my_funcs[i](y, y_hat))
		# print("real:", lst_sklearn_funcs[i](y, y_hat))

def zero_prediction_test():
	print("\nExample for zero_prediction_test")
	y_hat = np.zeros(8).reshape((-1, 1))
	y = np.array([1, 0, 0, 1, 0, 1, 0, 0]).reshape((-1, 1))
	#print("y_hat:", y_hat, y_hat.shape)

	lst_funcs_name = ["Accuracy", "Precision", "Recall", "F1-score"]
	lst_my_funcs = [accuracy_score_, precision_score_, recall_score_, f1_score_]
	lst_sklearn_funcs = [accuracy_score, precision_score, recall_score, f1_score]

	for i in range(len(lst_my_funcs)):
		print(lst_funcs_name[i])
		print("mine:", lst_my_funcs[i](y, y_hat))
		# print("real:", lst_sklearn_funcs[i](y, y_hat))

def invalid_label_test():
	print("\nExample for invalid_label_test")
	y_hat = np.array(['norminet', 'dog', 'norminet', 'norminet', 'dog', 'dog', 'dog', 'dog'])
	y = np.array(['dog', 'dog', 'norminet', 'norminet', 'dog', 'norminet', 'dog', 'norminet'])

	for i in range(1, len(lst_my_funcs)):
		print(lst_funcs_name[i])
		print("mine:", lst_my_funcs[i](y, y_hat, pos_label='none'))
	# 	print("real:", lst_sklearn_funcs[i](y, y_hat, pos_label='none'))

if __name__ == "__main__":
	ex1()
	ex2()
	ex3()
	# binary_continuous_target_test()
	# zero_prediction_test()
