import numpy as np
import pandas as pd
import sys, os, itertools, pickle
import matplotlib.pyplot as plt
from data_spliter import data_spliter

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
	print("y_labelled shape:", y_labelled.shape)
	#print("y_labelled[:5]:", y_labelled[:5])
	return y_labelled

def plot_logistic(x_features, y_hat):
	fig, axes = plt.subplots(1, 3, figsize=(30, 10))
	for i in range(len(x_features)):
		ax = axes[i]
		scatter = ax.scatter(x_test[:, i], y_test, c=y_hat, s=50, label="true value", cmap='viridis', vmin=0, vmax=1)
		ax.scatter(x_test[:, i], predictions, c='orange', s=7, label="prediction")
		ax.set_title(f'citizen\'s {x_features[i]} with Predictions')
		ax.set_xlabel(x_features[i])
		ax.legend()
		ax.grid()
		if i == len(x_features) - 1:
			cbar = plt.colorbar(scatter, ax=ax)
			cbar.set_label('Degree (0-1)')
	plt.tight_layout()
	plt.show()

def plot_scatter_of_each_pair(x_features, y_hat):
	feature_pairs = [(i, j) for i in range(len(x_features)) for j in range(i + 1, len(x_features))]

	for i, (f1, f2) in enumerate(feature_pairs):
		plt.scatter(x_test[:, f1], x_test[:, f2], c=y_hat, cmap='viridis', vmin=0, vmax=1)
		plt.title(f'Scatter Plot: {x_features[f1]} vs {x_features[f2]} with Predictions')
		plt.xlabel(x_features[f1])
		plt.ylabel(x_features[f2])
		plt.colorbar(label='Degree (0-1)')
		plt.show()

def plot_all_scatters_of_each_pair(x_features, y_hat):
	# Generate all possible combinations of feature pairs
	feature_pairs = [(i, j) for i in range(len(x_features)) for j in range(i + 1, len(x_features))]
	
	fig, axes = plt.subplots(len(feature_pairs), 1, figsize=(8, 6 * len(feature_pairs)))
	for i, (f1, f2) in enumerate(feature_pairs):
		ax = axes[i]
		ax.scatter(x_test[:, f1], x_test[:, f2], c=y_hat, cmap='viridis', vmin=0, vmax=1)
		ax.set_title(f"{x_features[f1]} vs {x_features[f2]}")
		ax.set_xlabel(x_features[f1])
		ax.set_ylabel(x_features[f2])
	plt.tight_layout()
	plt.show()

def data_spliter_by(x, y, zipcode):
	y_ = np.zeros(y.shape)
	y_[np.where(y == int(zipcode))] = 1
	y_labelled = y_.reshape(-1, 1)
	#print("y_labelled shape:", y_labelled.shape)
	#print("y_labelled[:5]:", y_labelled[:5])
	return data_spliter(x, y_labelled, 0.8)

def classify_citizen(classifiers, citizen, matching_planet):
	print(f"\nclassify citizen: {citizen}, real zipcode: {int(matching_planet)}")
	for zipcode in range(4):
		probability = classifiers[zipcode].predict_(citizen.reshape(1, -1))
		print(f"probability for zipcode {zipcode}: {float(probability) * 100:.2f}%")

def print_classifier_info(classifiers):
	for classifier in classifiers:
		print("Classifier:", classifier)
		print("Thetas:", classifier.thetas)
		print("Alpha:", classifier.alpha)
		print("Max Iterations:", classifier.max_iter)
		print("Lambda:", classifier.lambda_)
		print()

def load_models():
	# Load the models from the pickle file
	filename = "models.pickle"
	with open(filename, 'rb') as file:
		models = pickle.load(file)

	classifiers = []
	for zipcode, classifier_data in models.items():
		for key, value in classifier_data.items():
			if key == 'classifier':
				classifiers.append(value)
	
	print_classifier_info(classifiers)
	return models

def best_hypothesis(x_features, classifiers):
	x_train, x_test, y_train, y_test = data_spliter(x, y, 0.8)

	# 4. Determine the best hypothesis based on the evaluation metrics, similar to before.
	best_classifier = min(classifiers.items(), key=lambda x: x[1]['loss'])
	best_degree = best_classifier[0]

	# 5. Plot the evaluation curve, which shows the MSE for each model degree.
	degrees = list(classifiers.keys())
	mses = [classifiers[degree]['loss'] for degree in degrees]

	plt.plot(degrees, mses, marker='o')
	plt.xlabel('Degree')
	plt.ylabel('Mean Squared Error')
	plt.title('Model Evaluation Curve')
	plt.show()

	# 6. Train the best model on the entire training set using the selected degree.
	best_model = classifiers[best_degree]['model']
	# print("best_degree:", best_degree)
	x_train_poly = add_polynomial_features(x_train, best_degree)
	# print("x_train_poly:", x_train_poly.shape)
	best_model.fit_(x_train_poly, y_train)
	# 2. Train 4 logistic regression classifiers to discriminate each class from the others (the way you did in part one).
	# Train separate logistic regression classifiers
	predictions = np.zeros(y_test.shape)
	print(f"Starting training each classifier for logistic regression...")
	for zipcode, classifier in zip(range(4), classifiers):
		x_train, x_test, y_train_labelled, y_test_labelled = data_spliter_by(x, y, zipcode)

		probability = classifier.predict_(x_test)
		# 4. Calculate and display the fraction of correct predictions over the total number of predictions based on the test set.
		binary_predictions = (probability >= 0.5).astype(int)
		#print("binary_predictions:", binary_predictions[:5], binary_predictions.shape)
		predictions[np.where(binary_predictions == 1)] = zipcode
		correct_predictions = np.sum(binary_predictions == y_test_labelled)
		total_predictions = y_test_labelled.shape[0]
		fraction_correct = correct_predictions / total_predictions
		print(f"Fraction of correct predictions for classifier of zipcode {zipcode}: {correct_predictions}/{total_predictions} ({fraction_correct * 100:.2f}%)")
	predictions = predictions.reshape(-1, 1)
	#print("predictions:", predictions[:5], predictions.shape)

	#print_classifier_info(classifiers)
	# 3. Predict for each example the class according to each classifiers and select the one with the highest output probability.
	for citizen, planet in zip(x_test, y_test):
		classify_citizen(classifiers, citizen, planet)

	# 5. Plot 3 scatter plots (one for each pair of citizen features) with the dataset and the final prediction of the model.
	plot_logistic(x_features, probability)
	#plot_scatter_of_each_pair(x_features, probability)
	#plot_all_scatters_of_each_pair(x_features, probability)

if __name__ == "__main__":
	x, y, x_features = load_data()
	#print("normalized x\n:", x[:5], x.shape)
	#print("y\n:", y[:5], y.shape)

	classifiers = load_models()
	best_hypothesis(x_features, classifiers)
