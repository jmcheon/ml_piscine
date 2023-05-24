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

	if len(y) != len(y_hat):
		return None

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

	if len(y) != len(y_hat):
		return None

	true_positives = np.sum((y == pos_label) & (y_hat == pos_label))
	predicted_positives = np.sum(y_hat == pos_label)
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

	if len(y) != len(y_hat):
		return None

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

	if len(y) != len(y_hat):
		return None

	precision = precision_score_(y, y_hat, pos_label)
	recall = recall_score_(y, y_hat, pos_label)
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

if __name__ == "__main__":
	ex1()
	ex2()
	ex3()
