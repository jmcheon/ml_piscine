import sys, os, itertools, pickle
import numpy as np
import pandas as pd
from multiple_polynomial_model import add_polynomial_features

path = os.path.join(os.path.dirname(__file__), '..', 'ex05')
sys.path.insert(1, path)
from mylinearregression import MyLinearRegression as MyLR

path = os.path.join(os.path.dirname(__file__), '..', 'ex09')
sys.path.insert(1, path)
from data_spliter import data_spliter

def benchmark_train(degree, x_train_poly, x_test_poly, y_train, y_test):
	# Initialize variables to store the best hyperparameters and evaluation metric
	best_hyperparameters = {'thetas': None, 'alpha': None, 'max_iter': None}
	best_mse = float('inf')
	
	# Define the range of hyperparameter values to search
	thetas_range = [np.zeros((x_train_poly.shape[1] + 1, 1)), np.random.rand(x_train_poly.shape[1] + 1, 1)]
	alpha_range = [1e-4, 1e-3, 1e-2]
	max_iter_range = [10000, 50000, 100000]
	
	# Perform grid search to find the best hyperparameters
	for thetas, alpha, max_iter in itertools.product(thetas_range, alpha_range, max_iter_range):
		# Initialize and train the linear regression model
		model = MyLR(thetas, alpha, max_iter)
		model.fit_(x_train_poly, y_train)
		# Evaluate the model on the test set
		y_pred = model.predict_(x_test_poly)
		mse = model.mse_(y_test, y_pred)
		print(f"degree: {degree}, mse: {mse}, alpha: {alpha}, max_iter: {max_iter}")
	# Store the model and its evaluation metric
	dict_model = {'model': model, 'mse': mse}

	# Update the best hyperparameters if the current model performs better
	if mse < best_mse:
		best_mse = mse
		best_hyperparameters['thetas'] = model.thetas
		best_hyperparameters['alpha'] = alpha
		best_hyperparameters['max_iter'] = max_iter
	
	# Print the best hyperparameters and MSE
	print(f"Best Hyperparameters for degree {degree}:")
	print("Thetas:", best_hyperparameters['thetas'])
	print("Alpha:", best_hyperparameters['alpha'])
	print("Max Iterations:", best_hyperparameters['max_iter'])
	print("Best MSE:", best_mse)

	# Save the model as a pickle file
	#filename = f"model_degree_{degree}.pickle"
	#with open(filename, 'wb') as file:
        #	pickle.dump(dict_model, file)

	return dict_model


def benchmark(x_features):
	# 1. Perform min-max normalization by calculating the minimum and maximum values for each feature
	min_vals = np.min(data, axis=0)
	max_vals = np.max(data, axis=0)
	normalized_data = (data - min_vals) / (max_vals - min_vals)

	# 2. Split the space_avocado.csv dataset into a training set and a test set
	# Split the data into features (x) and target (y)
	x = np.array(normalized_data[x_features])
	y = np.array(normalized_data[['target']])
	print("x:", x.shape)
	print("y:", y.shape)
	x_train, x_test, y_train, y_test = data_spliter(x, y, 0.8)

	# 3. Consider several Linear Regression models with polynomial hypotheses of different degrees. 
	# Train and evaluate each model on the test set.
	# Initialize a dictionary to store the models and their evaluation metrics
	models = {}
	for degree in range(1, 5):
		# Create the polynomial features for the current degree
		x_train_poly = add_polynomial_features(x_train, degree)
		x_test_poly = add_polynomial_features(x_test, degree)
		models[degree] = benchmark_train(degree, x_train_poly, x_test_poly, y_train, y_test)

	# Save the models as a pickle file
	filename = "models.pickle"
	with open(filename, 'wb') as file:
        	pickle.dump(models, file)

if __name__ == "__main__":
	try:
		data = pd.read_csv("space_avocado.csv")
	except:
		print("Invalid file error.")
		sys.exit()
	x_features = ['weight', 'prod_distance', 'time_delivery']
	benchmark(x_features)
