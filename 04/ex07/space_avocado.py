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

def plot_scatters_and_prediction(x_features, x_test, y_test, best_model, best_degree):
	fig, axes = plt.subplots(1, 3, figsize=(20, 15))
	for i in range(len(x_features)):
		ax = axes[i]
		x_test_ploy = add_polynomial_features(x_test, best_degree)
		# print("x_test_ploy shape:", x_test_ploy.shape)

		# Make predictions using the best model
		y_pred = best_model.predict_(x_test_ploy)

		ax.scatter(x_test[:, i], y_test, color='blue', label='Data')
		ax.scatter(x_test[:, i], y_pred, color='red', label='Prediction')
		ax.set_xlabel(f'{x_features[i]}')
		ax.set_ylabel('Target')
		ax.set_title(f'Scatter plot and prediction for {x_features[i]}')
		ax.legend()
	plt.show()

def plot_3d(x_test, y_test, best_model, best_degree):
	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')

	# Generate predictions using the best model
	x_test_poly = add_polynomial_features(x_test, best_degree)
	y_pred = best_model.predict_(x_test_poly)

	# Plot the true price
	ax.scatter(x_test[:, 1], x_test[:, 2], y_test, c='b', label='True Price')
	# Plot the predicted price
	ax.scatter(x_test[:, 1], x_test[:, 2], y_pred, c='r', label='Predicted Price')
	ax.set_xlabel('prod_distance')
	ax.set_ylabel('time_delivery')
	ax.set_zlabel('Price')
	ax.legend()
	plt.show()

def normalization(data):
	data_min = np.min(data, axis=0)
	data_max = np.max(data, axis=0)
	normalized_data = (data - data_min) / (data_max - data_min)
	return normalized_data, data_min, data_max

def denormalization(normalized_data, data_min, data_max):
	denormalized_data = normalized_data * (data_max - data_min) + data_min
	return denormalized_data

def select_best_model(degrees, dict_best_model_by_degree, dict_best_mses_by_degree):
	best_model = None
	best_mse = float('inf')
	best_degree = 0
	best_difference = float('-inf')
	for degree in range(1, len(degrees)):
		difference = dict_best_mses_by_degree[degree] - dict_best_mses_by_degree[degree + 1]
		if difference > best_difference:
			best_difference = difference
			best_model = dict_best_model_by_degree[degree + 1]
			best_mse = dict_best_mses_by_degree[degree + 1]
			best_degree = degree + 1
		#print(f"best difference:{best_difference}")
		#print(f"difference:{difference}")
	print(f"Best degree: {best_degree}")
	print(f"Best mse: {best_mse}")
	return best_model, best_degree

def best_hypothesis(data, x_features, models):
	# 1. Perform min-max normalization by calculating the minimum and maximum values for each feature
	normalized_data, data_min, data_max = normalization(data)

	x = normalized_data[:, :len(x_features)]
	y = normalized_data[:, -1].reshape(-1, 1)
	print("x:", x.shape)
	print("y:", y.shape)
	# 2. Split your space_avocado.csv dataset into a training, a cross-validation and a test sets.  
	x_train, x_test, y_train, y_test = data_spliter(x, y, 0.8)
	x_train, x_validation, y_train, y_validation = data_spliter(x_train, y_train, 0.8) 
	# 3. Evaluate the best model on the test set.
	degrees = list(models.keys())
	dict_best_mses_by_degree = {}
	dict_best_model_by_degree = {}
	dict_mses = {}
	dict_lambdas = {}
	for degree, lst_models in models.items():
		best_model_by_degree = None
		best_mse_by_degree = float('inf')
		mses = []
		lambdas = []
		for i, model_data in zip(range(len(lst_models)), lst_models):
			model = model_data['model']
			x_test_poly = add_polynomial_features(x_test, degree)
			y_pred = model.predict_(x_test_poly)
			denormalized_y_test = denormalization(y_test.reshape(-1 ,1), data_min[-1], data_max[-1])
			denormalized_y_pred = denormalization(y_pred.reshape(-1, 1), data_min[-1], data_max[-1])

			#mse = model.mse_(denormalized_y_test, denormalized_y_pred)
			mse = model.mse_(y_test, y_pred)
			#print(degree, mse)
			mses.append(mse)
			lambdas.append(f"model{(degree - 1) * len(lst_models) + i + 1} + λ{model.lambda_:.1f}")
			if mse < best_mse_by_degree:
				best_model_by_degree = model
				best_mse_by_degree = mse
		dict_best_mses_by_degree[degree] = best_mse_by_degree
		dict_best_model_by_degree[degree] = best_model_by_degree
		dict_mses[degree] = mses
		dict_lambdas[degree] = lambdas 
	print(f"Best mses by degree: {dict_best_mses_by_degree}")
	#print("dict mses:", dict_mses)
	#print("dict_lambdas:", dict_lambdas)

	# 4. Plot the evaluation curve which help you to select the best model (evaluation metrics vs models + λ factor).
	fig, axes = plt.subplots(2, 1, figsize=(15, 15))
	fig.tight_layout(pad=15)
	colors = ['red', 'green', 'blue', 'orange']
	for i in range(len(degrees)):
		ax = axes[1]
		if i < len(degrees) // 2:
			ax = axes[0]
		ax.scatter(dict_lambdas[i + 1], dict_mses[i + 1], c=colors[i], label=f"degree {i + 1}")
		ax.set_xlabel("λ Values")
		ax.set_ylabel("MSE")
		ax.set_title("Evaluation Curve: MSE vs. models(λ Values)")
		ax.legend()
		ax.grid()
	plt.setp(axes[0].get_xticklabels(), rotation=90)
	plt.setp(axes[1].get_xticklabels(), rotation=90)
	plt.show()
	
	# 5. Determine the best hypothesis based on the evaluation metrics, similar to before.
	lst_degree_lambda_tags = []
	for degree, model in zip(degrees, dict_best_model_by_degree.values()):
		lst_degree_lambda_tags.append(f"degree {degree} + λ {model.lambda_}")
	plt.plot(lst_degree_lambda_tags, dict_best_mses_by_degree.values(), marker='o')
	plt.xlabel('Degree + λ')
	plt.ylabel('Mean Squared Error')
	plt.title('Model Evaluation Curve')
	plt.show()
	best_model, best_degree = select_best_model(degrees, dict_best_model_by_degree, dict_best_mses_by_degree)

	# 6. Train the best model on the entire training set using the selected degree.
	print("Training the best model on the entire training set using the selected degree....")
	x_train_poly = add_polynomial_features(x_train, best_degree)
	# print("x_train_poly:", x_train_poly.shape)
	best_model.fit_(x_train_poly, y_train)

	# 6. Plot the true price and the predicted price obtain via your best model with the different λ values 
	# (meaning the dataset + the 5 predicted curves).
	plot_3d(x_test, y_test, best_model, best_degree)
	plot_scatters_and_prediction(x_features, x_test, y_test, best_model, best_degree)

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
		data = pd.read_csv("space_avocado.csv", index_col=0)
	except:
		print("Invalid file error.")
		sys.exit()
	x_features = data.columns.tolist()[:-1] #['weight', 'prod_distance', 'time_delivery']
	#plot_scatters(x_features)
	models = load_models()
	print("x features:", x_features)
	best_hypothesis(data.values, x_features, models)
