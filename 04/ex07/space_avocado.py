import sys, os, pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from data_spliter import data_spliter

path = os.path.join(os.path.dirname(__file__), '..', 'ex00')
sys.path.insert(1, path)
from polynomial_model_extended import add_polynomial_features

path = os.path.join(os.path.dirname(__file__), '..', 'ex06')
sys.path.insert(1, path)
from ridge import MyRidge

def plot_scatters(x_features):
	X = []
	for i in range(len(x_features)):
		X.append(np.array(data[[x_features[i]]]))
	Y = np.array(data[['target']])
	_, axes = plt.subplots(1, 3, figsize=(15, 8))
	for i in range(len(x_features)):
		axes[i].scatter(X[i], Y, label='raw', c='black')
		axes[i].set_xlabel(x_features[i])
	axes[0].set_ylabel('target')
	plt.grid()
	plt.legend()
	plt.show()

def plot_scatters_and_prediction(x_features, x, y ,best_model):
	# Iterate over each feature
	for i in range(len(x_features)):
		# Extract the feature values and corresponding target values
		x_feature = x[:, i].reshape(-1, 1)
		y_target = y[:, 0].reshape(-1, 1)
		x_train, x_test, y_train, y_test = data_spliter(x_feature, y_target, 0.8)
		print("feature:", x_features[i])
		print("x_feature:" ,x_feature.shape)
		print("y_target:", y_target.shape)

		# Reshape the sorted feature values to match the number of columns in the thetas
		x_feature_ploy = add_polynomial_features(x_test, best_model.thetas.shape[0] - 1)
		# print("x_feature_ploy shape:", x_feature_ploy.shape)

		# Make predictions using the best model
		y_pred = best_model.predict_(x_feature_ploy)

		# Plot the scatter plot and prediction line
		plt.scatter(x_test, y_test, color='blue', label='Data')
		plt.plot(x_test, y_pred, color='red', label='Prediction')
		plt.xlabel(f'{x_features[i]}')
		plt.ylabel('Target')
		plt.title(f'Scatter plot and prediction for {x_features[i]}')
		plt.legend()
		plt.show()

def normalization(data):
	data_min = np.min(data, axis=0)
	data_max = np.max(data, axis=0)
	normalized_data = (data - data_min) / (data_max - data_min)
	return normalized_data, data_min, data_max

def denormalization(normalized_data, data_min, data_max):
	denormalized_data = normalized_data * (data_max - data_min) + data_min
	return denormalized_data

def best_hypothesis(x_features, models):
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


	# 3. Evaluate the best model on the test set.
	best_model = None
	best_mse = float('inf')
	best_degree = 0
	dict_mses = {}
	dict_lambdas = {}
	for degree, lst_models in models.items():
		mses = []
		lambdas = []
		for model_data in lst_models:
			model = model_data['model']
			x_test_poly = add_polynomial_features(x_test, degree)
			y_pred = model.predict_(x_test_poly)
			#y_pred = model.predict_(x_test)
			mse = model.mse_(y_test, y_pred)
			print(degree, mse)
			mses.append(mse)
			lambdas.append(model.lambda_)
			if mse < best_mse:
				best_model = model
				best_mse = mse
				best_degree = degree
		dict_mses[degree] = mses
		dict_lambdas[degree] = lambdas 
	print(f"Best mse with test set: {best_mse}")
	print("dict mses:", dict_mses)
	print("dict_lambdas:", dict_lambdas)

	# 4. Determine the best hypothesis based on the evaluation metrics, similar to before.
	#best_model = min(models.items(), key=lambda x: x[1]['mse'])
	#best_degree = best_model[0]
	degrees = list(models.keys())
	#best_model = models[best_degree]['model']
	print("best_degree:", best_degree)

	# 5. Plot the evaluation curve which help you to select the best model (evaluation metrics vs models + λ factor).
	#mses = [models[degree]['mse'] for degree in degrees]
	#lambda_values = [model_data['model'].lambda_ for degree, model_data in models.items()]
	lambda_values = [model_data['model'].lambda_ for model_data in lst_models for degree, lst_models in models.items()]
	lst_models = []
	for i, lambda_ in zip(range(0, len(lambda_values)), lambda_values):
		lst_models.append(f"{i + 1}")
	
	for i in range(len(degrees)):
		plt.plot(dict_lambdas[i + 1], dict_mses[i + 1], marker='o', label=f"degree {i}")
	plt.xlabel("λ Values")
	plt.ylabel("MSE")
	plt.title("Evaluation Curve: MSE vs. models(λ Values)")
	#ax[1].plot(degrees, lambda_values)
	#ax[1].set_xlabel("Degree")
	#ax[1].set_ylabel("λ Values")
	#ax[1].set_title("λ Values vs models(degree)")
	plt.legend()
	plt.grid()
	plt.show()

	# 6. Plot the true price and the predicted price obtain via your best model with the different λ values 
	# (meaning the dataset + the 5 predicted curves).
	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')

	# Plot the true price
	ax.scatter(x_test[:, 1], x_test[:, 2], y_test, c='b', label='True Price')

	# Generate predictions using the best model
	x_test_poly = add_polynomial_features(x_test, best_degree)
	# print("x_test_poly:", x_test_poly.shape)
	y_pred = best_model.predict_(x_test_poly)

	# Plot the predicted price
	ax.scatter(x_test[:, 1], x_test[:, 2], y_pred, c='r', label='Predicted Price')
	ax.set_xlabel('prod_distance')
	ax.set_ylabel('time_delivery')
	ax.set_zlabel('Price')
	ax.legend()
	plt.show()
	#plot_scatters_and_prediction(x_features, x, y, best_model)

def load_model_by_degree(degree):
	# Load the models from the pickle file
	filename = f"model_degree_{degree}.pickle"
	with open(filename, 'rb') as file:
		model = pickle.load(file)
	
	print(f"Degree: {degree}")
	#print("Model:", model['model'])
	print("Thetas:", model['model'].thetas)
	print("Alpha:", model['model'].alpha)
	print("Max Iterations:", model['model'].max_iter)
	print("Lambda:", model_data['model'].lambda_)
	print("MSE:", model['mse'])
	print()

def load_models():
	# Load the models from the pickle file
	filename = "models.pickle"
	with open(filename, 'rb') as file:
		models = pickle.load(file)
	
	# Iterate and print the data
	for degree, lst_models in models.items():
		print(f"Degree: {degree}")
		for model_data in lst_models:
			#print("Model:", model_data)
			print("Thetas:", model_data['model'].thetas)
			print("Alpha:", model_data['model'].alpha)
			print("Max Iterations:", model_data['model'].max_iter)
			print("Lambda:", model_data['model'].lambda_)
			print("MSE:", model_data['mse'])
			print()
	return models

if __name__ == "__main__":
	try:
		data = pd.read_csv("space_avocado.csv")
	except:
		print("Invalid file error.")
		sys.exit()
	x_features = ['weight', 'prod_distance', 'time_delivery']
	#plot_scatters(x_features)
	models = load_models()
	best_hypothesis(x_features, models)
