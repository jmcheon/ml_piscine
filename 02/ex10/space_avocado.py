import sys, os, pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from multiple_polynomial_model import add_polynomial_features

path = os.path.join(os.path.dirname(__file__), '..', 'ex05')
sys.path.insert(1, path)
from mylinearregression import MyLinearRegression as MyLR

path = os.path.join(os.path.dirname(__file__), '..', 'ex09')
sys.path.insert(1, path)
from data_spliter import data_spliter

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
		print("feature:", x_features[i])
		print("x_feature:" ,x_feature.shape)
		print("y_target:", y_target.shape)

		# Reshape the sorted feature values to match the number of columns in the thetas
		x_feature_ploy = add_polynomial_features(x_feature, best_model.thetas.shape[0] - 1)
		# print("x_feature_ploy shape:", x_feature_ploy.shape)

		# Make predictions using the best model
		y_pred = best_model.predict_(x_feature_ploy)

		# Plot the scatter plot and prediction line
		plt.scatter(x_feature, y_target, color='blue', label='Data')
		plt.plot(x_feature, y_pred, color='red', label='Prediction')
		plt.xlabel(f'{x_features[i]}')
		plt.ylabel('Target')
		plt.title(f'Scatter plot and prediction for {x_features[i]}')
		plt.legend()
		plt.show()

def best_hypothesis(x_features, models):
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

	# 4. Determine the best hypothesis based on the evaluation metrics, similar to before.
	best_model = min(models.items(), key=lambda x: x[1]['mse'])
	best_degree = best_model[0]

	# 5. Plot the evaluation curve, which shows the MSE for each model degree.
	degrees = list(models.keys())
	mses = [models[degree]['mse'] for degree in degrees]

	plt.plot(degrees, mses, marker='o')
	plt.xlabel('Degree')
	plt.ylabel('Mean Squared Error')
	plt.title('Model Evaluation Curve')
	plt.show()

	# 6. Train the best model on the entire training set using the selected degree.
	best_model = models[best_degree]['model']
	# print("best_degree:", best_degree)
	x_train_poly = add_polynomial_features(x_train, best_degree)
	# print("x_train_poly:", x_train_poly.shape)
	best_model.fit_(x_train_poly, y_train)

	# 7. Plot the true price and the predicted price obtained via the best model. 
	# You can create a 3D scatter plot or separate scatter plots for each feature against the target variable.
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
	print("MSE:", model['mse'])
	print()

def load_models():
	# Load the models from the pickle file
	filename = "models.pickle"
	with open(filename, 'rb') as file:
		models = pickle.load(file)
	
	# Iterate and print the data
	for degree, model_data in models.items():
		print(f"Degree: {degree}")
		#print("Model:", model_data)
		print("Thetas:", model_data['model'].thetas)
		print("Alpha:", model_data['model'].alpha)
		print("Max Iterations:", model_data['model'].max_iter)
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
