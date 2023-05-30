import numpy as np
import pandas as pd
import sys, os, itertools, pickle
import matplotlib.pyplot as plt
from other_metrics import f1_score_
from data_spliter import data_spliter

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
	print("y_labelled shape:", y_labelled.shape)
	#print("y_labelled[:5]:", y_labelled[:5])
	return y_labelled

def plot_logistic(x_features, x_test, y_test, y_hat, predictions):
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

def plot_scatter_of_each_pair(x_features, x_test, y_test, y_hat, predictions):
	feature_pairs = [(i, j) for i in range(len(x_features)) for j in range(i + 1, len(x_features))]

	for i, (f1, f2) in enumerate(feature_pairs):
		plt.scatter(x_test[:, f1], x_test[:, f2], c=y_hat, cmap='viridis', vmin=0, vmax=1)
		plt.title(f'Scatter Plot: {x_features[f1]} vs {x_features[f2]} with Predictions')
		plt.xlabel(x_features[f1])
		plt.ylabel(x_features[f2])
		plt.colorbar(label='Degree (0-1)')
		plt.show()

def plot_all_scatters_of_each_pair(x_features, x_test, y_test, y_hat, predictions):
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

def classify_citizen(classifiers, degrees, citizen, matching_planet):
	print(f"\nclassify citizen: {citizen}, real zipcode: {int(matching_planet)}")
	for zipcode in range(4):
		citizen_poly = add_polynomial_features(citizen.reshape(1, -1), degrees[zipcode])
		probability = classifiers[zipcode].predict_(citizen_poly)
		print(f"probability for zipcode {zipcode}: {float(probability) * 100:.2f}%")

def print_classifier_info(classifiers):
	for classifier in classifiers:
		#print("Classifier:", classifier)
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
		print(f"\nzipcode:{zipcode}")
		for key, value in classifier_data.items():
			if key == 'classifier':
				print("Thetas:", value.thetas)
				print("Alpha:", value.alpha)
				print("Max Iterations:", value.max_iter)
				print("Lambda:", value.lambda_)
				classifiers.append(value)
			else:
				print(f"{key}:{value}")
	
	#print_classifier_info(classifiers)
	return models

def get_classifiers(models):
	classifiers = []
	degrees = []
	zipcodes = []
	for zipcode, classifier_data in models.items():
		zipcodes.append(zipcode)
		for key, value in classifier_data.items():
			if key == 'classifier':
				classifiers.append(value)
			elif key == 'degree':
				degrees.append(value)
	print("classifiers:", classifiers)
	print("degrees:", degrees)
	print("zipcodes:", zipcodes)
	print("zip:", zip(classifiers, degrees, zipcodes))
	return zip(classifiers, degrees)

def best_hypothesis(x_features, models):
	x_train, x_test, y_train, y_test = data_spliter(x, y, 0.8)

	# Determine the best hypothesis based on the evaluation metrics, similar to before.
	best_dict_classifier = max(models.items(), key=lambda x: x[1]['f1_score'])
	#print(f"best_dict_classifier:{best_dict_classifier}")
	best_zipcode = best_dict_classifier[0]
	#print(f"best_zipcode:{best_zipcode}")
	best_degree =  best_dict_classifier[1]['degree']
	#print(f"best_degree:{best_degree}")

	# Train from scratch only the best one on a training set.
	#best_classifier = models[best_zipcode]['classifier']
	#x_train_poly = add_polynomial_features(x_train, best_degree)
	#best_classifier.fit_(x_train_poly, y_train)

	# Visualize the performance of the different models with a bar plot showing the score of the models given their λ value.
	zipcodes = list(models.keys())
	f1_score_values = [models[zipcode]['f1_score'] for zipcode in zipcodes]

	plt.bar(zipcodes, f1_score_values)
	plt.xlabel('zipcode')
	plt.ylabel('f1 score')
	plt.title('Model Evaluation Curve')
	plt.show()


	# Print the f1 score of all the models calculated on the test set
	predictions = np.zeros(y_test.shape)
	all_predictions = np.zeros(y_test.shape)
	classifiers = {}
	degrees = []
	print(f"Starting training each classifier for logistic regression...")
	for zipcode, classifier_data in models.items():
		#print(classifier_data)
		classifier = classifier_data['classifier']
		classifiers[zipcode] = classifier
		degree = classifier_data['degree']
		f1_score_value = classifier_data['f1_score']
		degrees.append(degree)

		x_train, x_test, y_train_labelled, y_test_labelled = data_spliter_by(x, y, zipcode)
		x_test_poly = add_polynomial_features(x_test, degree)

		probability = classifier.predict_(x_test_poly)
		binary_predictions = (probability >= 0.5).astype(int)
		#print("binary_predictions:", binary_predictions[:5], binary_predictions.shape)
		all_predictions[np.where(binary_predictions == 1)] = zipcode
		predictions[np.where(binary_predictions == 1)] = 1
		#f1_score_value = f1_score_(y_test_labelled, predictions)
		print(f"f1 score on test set for zipcode {zipcode}: {f1_score_value}")
	#print("predictions:", predictions[:5], predictions.shape)

	# Predict for each example the class according to each classifiers and select the one with the highest output probability.
	#for citizen, planet in zip(x_test, y_test):
	#	classify_citizen(classifiers, degrees, citizen, planet)

	# Visualize the target values and the predicted values of the best model on the same scatterplot. Make some effort to have a readable figure.
	plot_logistic(x_features, x_test, y_test, probability, all_predictions)
	#plot_scatter_of_each_pair(x_features, x_test, y_test, probability, all_predictions)
	#plot_all_scatters_of_each_pair(x_features, x_test, y_test, probability, all_predictions)

if __name__ == "__main__":
	x, y, x_features = load_data()

	models = load_models()
	best_hypothesis(x_features, models)
