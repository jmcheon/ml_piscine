import numpy as np
import pandas as pd
import sys, os, itertools, pickle
from data_spliter import data_spliter
from other_metrics import f1_score_

path = os.path.join(os.path.dirname(__file__), '..', 'ex00')
sys.path.insert(1, path)
from polynomial_model_extended import add_polynomial_features

path = os.path.join(os.path.dirname(__file__), '..', 'ex08')
sys.path.insert(1, path)
from my_logistic_regression import MyLogisticRegression as MyLR

def load_data():
	try:
		x = pd.read_csv("solar_system_census.csv", index_col=0)
		y = pd.read_csv("solar_system_census_planets.csv", index_col=0)
	except:
		print("Invalid file error.")
		sys.exit()
	print("x shape:", x.shape)
	print("y shape:", y.shape)
	if x.shape[0] != y.shape[0]:
		print(f"Invalid input: the number of rows should be matched between {x.shape} and {y.shape}")
		return None
	x_features = x.columns.tolist()
	print("x features:", x_features)

	# Normalization
	x_min = np.min(x, axis=0)
	x_max = np.max(x, axis=0)
	normalized_x = (x - x_min) / (x_max - x_min)
	return (normalized_x.values, y.values, x_features)

def label_data(x, y, zipcode):
	y_ = np.zeros(y.shape)
	y_[np.where(y == int(zipcode))] = 1
	y_labelled = y_.reshape(-1, 1)
	#print("y_labelled shape:", y_labelled.shape)
	#print("y_labelled[:5]:", y_labelled[:5])
	return y_labelled

def data_spliter_by(x, y, zipcode):
	y_ = np.zeros(y.shape)
	y_[np.where(y == int(zipcode))] = 1
	y_labelled = y_.reshape(-1, 1)
	#print("y_labelled shape:", y_labelled.shape)
	#print("y_labelled[:5]:", y_labelled[:5])
	return data_spliter(x, y_labelled, 0.8)

def benchmark_train(degree, zipcode, x_train_poly, x_validation_poly, y_train, y_validation):
	# Initialize variables to store the best hyperparameters and evaluation metric
	best_classifier = None
	best_hyperparameters = {'thetas': None, 'alpha': None, 'max_iter': None, 'lambda': None}
	best_f1_score = -1
	best_loss = float('inf')
	
	# Define the range of hyperparameter values to search
	thetas_range = [np.zeros((x_train_poly.shape[1] + 1, 1)), np.random.rand(x_train_poly.shape[1] + 1, 1)]
	alpha_range = [1e-2, 1e-1]
	max_iter_range = [10000, 50000]
	lambda_range = np.linspace(0.0, 1.0, num=3)
	
	# Perform grid search to find the best hyperparameters
	predictions = np.zeros(y_validation.shape)
	for thetas, alpha, max_iter, lambda_ in itertools.product(thetas_range, alpha_range, max_iter_range, lambda_range):
		# Initialize and train the logistic regression classifier
		classifier = MyLR(thetas, alpha, max_iter, lambda_=lambda_)
		classifier.fit_(x_train_poly, y_train)

		# Evaluate the classifier on the test set
		#print(x_validation_poly[:5])
		probability = classifier.predict_(x_validation_poly)
		#print(probability[:5])
		binary_predictions = (probability >= 0.5).astype(int)
		predictions[np.where(binary_predictions == 1)] = 1
		loss = classifier.loss_(y_validation, probability)
		f1_score_value = f1_score_(y_validation, predictions)

		print(f"zipcode: {zipcode}, degree: {degree}, loss: {loss}, f1_score: {f1_score_value}, alpha: {alpha}, max_iter: {max_iter}, lambda: {lambda_}")

		# Update the best hyperparameters if the current classifier performs better
		if f1_score_value > best_f1_score and loss < best_loss:
			best_classifier = classifier
			best_loss = loss 
			best_f1_score = f1_score_value
			best_hyperparameters['thetas'] = classifier.thetas
			best_hyperparameters['alpha'] = alpha
			best_hyperparameters['max_iter'] = max_iter
			best_hyperparameters['lambda'] = lambda_
	# Store the classifier and its evaluation metric
	dict_classifier = {'classifier': best_classifier, 'loss': best_loss, 'f1_score': best_f1_score, 'degree': degree}

	# Print the best hyperparameters and f1 score
	print(f"Best Hyperparameters for zipcode {zipcode}:")
	print("Thetas:", best_hyperparameters['thetas'])
	print("Alpha:", best_hyperparameters['alpha'])
	print("Max Iterations:", best_hyperparameters['max_iter'])
	print("Lambda:", best_hyperparameters['lambda'])
	print("Best f1 score:", best_f1_score)

	return dict_classifier

def benchmark(x_features):
	print(f"Starting training each classifier for logistic regression...")
	classifiers = {}
	for zipcode in range(4):
		print(f"Current zipcode: {zipcode}")
		x_train, x_test, y_train, y_test = data_spliter_by(x, y, zipcode)
		x_train, x_validation, y_train, y_validation = data_spliter_by(x_train, y_train, zipcode)
		for degree in range(1, 4):
			print(f"Current degree: {degree}")
			# Create the polynomial features for the current degree
			x_train_poly = add_polynomial_features(x_train, degree)
			x_validation_poly = add_polynomial_features(x_validation, degree)
			classifiers[zipcode] = benchmark_train(degree, zipcode, x_train_poly, x_validation_poly, y_train, y_validation)

	filename = "models.pickle"
	with open(filename, 'wb') as file:
        	pickle.dump(classifiers, file)

if __name__ == "__main__":
	x, y, x_features = load_data()
	benchmark(x_features)
