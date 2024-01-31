import sys, os, itertools, pickle
import numpy as np
import pandas as pd
from data_spliter import data_spliter

path = os.path.join(os.path.dirname(__file__), '..', 'ex00')
sys.path.insert(1, path)
from polynomial_model_extended import add_polynomial_features

path = os.path.join(os.path.dirname(__file__), '..', 'ex06')
sys.path.insert(1, path)
from ridge import MyRidge

def benchmark_train(degree, x_train_poly, x_validation_poly, y_train, y_validation):
	# Define the range of hyperparameter values to search
	thetas_range = [np.zeros((x_train_poly.shape[1] + 1, 1)), np.random.rand(x_train_poly.shape[1] + 1, 1)]
	alpha_range = [1e-2]
	max_iter_range = [50000]
	lambda_range = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0] #np.linspace(0.0, 1.0, num=6)
	
	models = [] 
	# Perform grid search to find the best hyperparameters
	for thetas, alpha, max_iter, lambda_  in itertools.product(thetas_range, alpha_range, max_iter_range, lambda_range):
		# Initialize and train the linear regression model
		model = MyRidge(thetas, alpha, max_iter, lambda_)
		model.fit_(x_train_poly, y_train)
		# Evaluate the model on the cross-validation set
		y_pred = model.predict_(x_validation_poly)
		mse = model.mse_(y_validation, y_pred)
		print(f"degree: {degree}, mse: {mse}, alpha: {alpha}, max_iter: {max_iter}, lambda_: {lambda_}")
		models.append({'model': model, 'mse': mse})
	return models 


def benchmark(x_features):
	# 1. Perform min-max normalization by calculating the minimum and maximum values for each feature
	min_vals = np.min(data, axis=0)
	max_vals = np.max(data, axis=0)
	normalized_data = (data - min_vals) / (max_vals - min_vals)

	x = np.array(normalized_data[x_features])
	y = np.array(normalized_data[['target']])
	print("x:", x.shape)
	print("y:", y.shape)

	# 2. Split your space_avocado.csv dataset into a training, a cross-validation and a test sets.
	x_train, x_test, y_train, y_test = data_spliter(x, y, 0.8)
	x_train, x_validation, y_train, y_validation = data_spliter(x_train, y_train, 0.8)

	# 3. Consider several Linear Regression models with polynomial hypotheses of different degrees of 4
	# Initialize a dictionary to store the models and their evaluation metrics
	models = {}
	for degree in range(1, 5):
		# Create the polynomial features for the current degree
		x_train_poly = add_polynomial_features(x_train, degree)
		x_validation_poly = add_polynomial_features(x_validation, degree)
		models[degree] = benchmark_train(degree, x_train_poly, x_validation_poly, y_train, y_validation)

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
