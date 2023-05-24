import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

def confusion_matrix_(y_true, y_hat, labels=None, df_option=False):
	"""
	Compute confusion matrix to evaluate the accuracy of a classification.

	Args:
	y:a numpy.array for the correct labels
	y_hat:a numpy.array for the predicted labels
	labels: optional, a list of labels to index the matrix.
	This may be used to reorder or select a subset of labels. (default=None)
	df_option: optional, if set to True the function will return a pandas DataFrame
	instead of a numpy array. (default=False)

	Return:
	The confusion matrix as a numpy array or a pandas DataFrame according to df_option value.
	None if any error.

	Raises:
	This function should not raise any Exception.
	"""
	# Get unique labels from true and predicted labels
	unique_labels = np.unique(np.concatenate((y_true, y_hat)))
	
	# If labels are provided, use them for row and column indices
	if labels is not None:
		unique_labels = np.array(labels)
	
	num_labels = len(unique_labels)
	confusion_matrix = np.zeros((num_labels, num_labels), dtype=np.int64)
	
	# Fill the confusion matrix
	for i in range(num_labels):
		for j in range(num_labels):
			true_mask = (y_true == unique_labels[i])
			pred_mask = (y_hat == unique_labels[j])
			confusion_matrix[i, j] = np.sum(true_mask & pred_mask)
	
	# Return the confusion matrix as numpy array or pandas DataFrame
	if df_option:
		return pd.DataFrame(confusion_matrix, index=unique_labels, columns=unique_labels)
	else:
		return confusion_matrix

def ex1(df_option=False):	
	y_hat = np.array([['norminet'], ['dog'], ['norminet'], ['norminet'], ['dog'], ['bird']])
	y = np.array([['dog'], ['dog'], ['norminet'], ['norminet'], ['dog'], ['norminet']])
	print("Example 1")
	## your implementation
	print("mine:", confusion_matrix_(y, y_hat)) ## Output: array([[0 0 0] [0 2 1] [1 0 2]])
	## sklearn implementation
	print("real:", confusion_matrix(y, y_hat)) ## Output: array([[0 0 0] [0 2 1] [1 0 2]])

	print("Example 2")
	## your implementation
	print("mine:", confusion_matrix_(y, y_hat, labels=['dog', 'norminet'])) ## Output: array([[2 1] [0 2]])
	## sklearn implementation
	print("real:", confusion_matrix(y, y_hat, labels=['dog', 'norminet'])) ## Output: array([[2 1] [0 2]])
	if df_option:
		print("mine:", confusion_matrix_(y, y_hat, df_option=True))
		print("mine:", confusion_matrix_(y, y_hat, labels=['bird', 'dog'], df_option=True))

if __name__ == "__main__":
	ex1()
