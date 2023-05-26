import numpy as np
import pandas as pd
import sys, os
import matplotlib.pyplot as plt
from data_spliter import data_spliter

path = os.path.join(os.path.dirname(__file__), '..', 'ex06')
sys.path.insert(1, path)
from my_logistic_regression import MyLogisticRegression as MyLR

def display_usage(program_name):
	print(f"Usage: {program_name} -zipcode=x")
	print("       where x can be 0, 1, 2, or 3")

def parse_zipcode():
	zipcode = None
	if len(sys.argv) != 2:
		display_usage(sys.argv[0])
	else:
		arg = sys.argv[1]
		if arg.startswith("-zipcode="):
			zipcode = arg.split('=')[1]
			if zipcode in ['0', '1', '2', '3']:
				print("Processing zipcode", zipcode)
			else:
				print("Invalid zipcode: Valid values are 0, 1, 2, or 3.")
				zipcode = None
				display_usage(sys.argv[0])
		else:
			print("Invalid argument: Please provide the -zipcode argument.")
			display_usage(sys.argv[0])
	return zipcode

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

def data_spliter_by(x, y, zipcode):
	y_ = np.zeros(y.shape)
	y_[np.where(y == int(zipcode))] = 1
	y_labelled = y_.reshape(-1, 1)
	#print("y_labelled shape:", y_labelled.shape)
	#print("y_labelled[:5]:", y_labelled[:5])
	return data_spliter(x, y_labelled, 0.8)

def plot_logistic_with_scatter(x_features):
	for i in range(len(x_features)):
		fig, axes = plt.subplots(1, 3, figsize=(30, 10))
		for j in range(len(x_features)):
			ax = axes[j]
			ax.scatter(x_test[:, i], x_test[:, j], c=y_hat, s=50, cmap='viridis', vmin=0, vmax=1)
			ax.set_title(f'{x_features[i]} x {x_features[j]} with Predictions')
			ax.set_xlabel(x_features[i])
			scatter = ax.scatter(x_test[:, i], y_test, c=y_hat, s=50, label="true value", cmap='viridis', vmin=0, vmax=1)
			ax.scatter(x_test[:, i], binary_predictions, c='orange', s=7, label="prediction")
			ax.legend()
			ax.grid()
			if j == len(x_features) - 1:
				cbar = plt.colorbar(scatter, ax=ax)
				cbar.set_label('Degree (0-1)')
		plt.tight_layout()
		plt.show()

def plot_logistic(x_features):
	fig, axes = plt.subplots(1, 3, figsize=(30, 10))
	for i in range(len(x_features)):
		ax = axes[i]
		scatter = ax.scatter(x_test[:, i], y_test, c=y_hat, s=50, label="true value", cmap='viridis', vmin=0, vmax=1)
		ax.scatter(x_test[:, i], binary_predictions, c='orange', s=7, label="prediction")
		ax.set_title(f'citizen\'s {x_features[i]} with Predictions')
		ax.set_xlabel(x_features[i])
		ax.legend()
		ax.grid()
		if i == len(x_features) - 1:
			cbar = plt.colorbar(scatter, ax=ax)
			cbar.set_label('Degree (0-1)')
	plt.tight_layout()
	plt.show()

def plot_scatter_of_each_pair(x_features):
	feature_pairs = [(i, j) for i in range(len(x_features)) for j in range(i + 1, len(x_features))]

	for i, (f1, f2) in enumerate(feature_pairs):
		plt.scatter(x_test[:, f1], x_test[:, f2], c=y_hat, cmap='viridis', vmin=0, vmax=1)
		plt.title(f'{x_features[f1]} vs {x_features[f2]} with Predictions')
		plt.xlabel(x_features[f1])
		plt.ylabel(x_features[f2])
		plt.colorbar(label='Degree (0-1)')
		plt.show()

def plot_all_scatters_of_each_pair(x_features):
	# Generate all possible combinations of feature pairs
	feature_pairs = [(i, j) for i in range(len(x_features)) for j in range(i + 1, len(x_features))]
	
	fig, axes = plt.subplots(len(feature_pairs), 1, figsize=(8, 6 * len(feature_pairs)))
	for i, (f1, f2) in enumerate(feature_pairs):
		ax = axes[i]
		ax.scatter(x_test[:, f1], x_test[:, f2], c=y_hat, cmap='viridis')
		ax.set_title(f"{x_features[f1]} vs {x_features[f2]}")
		ax.set_xlabel(x_features[f1])
		ax.set_ylabel(x_features[f2])
	plt.tight_layout()
	plt.show()

if __name__ == "__main__":
	# 1. Take an argument: –zipcode=x with x being 0, 1, 2 or 3. If no argument, usage will be displayed.
	zipcode = parse_zipcode()
	if zipcode == None:
		sys.exit()

	# 2. Split the dataset into a training and a test set.
	x, y, x_features = load_data()
	#print("normalized x\n:", x[:5], x.shape)
	#print("y\n:", y[:5], y.shape)

	# 3. Select your favorite Space Zipcode and generate a new numpy.array to label each
	# citizen according to your new selection criterion: 1 or 0
	x_train, x_test, y_train, y_test = data_spliter_by(x, y, zipcode)

	# 4. Train a logistic model to predict if a citizen comes from your favorite planet or not, using your brand new label.
	thetas = np.random.rand(x.shape[1] + 1, 1)
	classifier = MyLR(thetas, 1e-2, 100000)
	classifier.fit_(x_train, y_train)
	y_hat = classifier.predict_(x_test)
	print(f"\nHyperparameters:")
	print("thetas (original):", thetas)
	print("thetas (optimized):", classifier.thetas)
	print("loss:", classifier.loss_(y_test, y_hat))

	# 5. Calculate and display the fraction of correct predictions over the total number of predictions based on the test set.
	threshold = 0.5
	# Convert probabilities to binary predictions
	binary_predictions = (y_hat >= threshold).astype(int)
	correct_predictions = np.sum(binary_predictions == y_test)
	total_predictions = y_test.shape[0]
	fraction_correct = correct_predictions / total_predictions
	print(f"\nFraction of correct predictions: {correct_predictions}/{total_predictions} ({fraction_correct}%)")

	# 6. Plot 3 scatter plots (one for each pair of citizen features) with the dataset and the final prediction of the model.
	plot_logistic(x_features)
	plot_logistic_with_scatter(x_features)
	plot_scatter_of_each_pair(x_features)
	plot_all_scatters_of_each_pair(x_features)
