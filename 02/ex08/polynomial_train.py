import sys, os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

path = os.path.join(os.path.dirname(__file__), '..', 'ex05')
sys.path.insert(1, path)
from mylinearregression import MyLinearRegression as MyLR

path = os.path.join(os.path.dirname(__file__), '..', 'ex07')
sys.path.insert(1, path)
from polynomial_model import add_polynomial_features

def ex1():
	x = np.arange(1,11).reshape(-1,1)
	y = np.array([[ 1.39270298],
			[ 3.88237651],
			[ 4.37726357],
			[ 4.63389049],
			[ 7.79814439],
			[ 6.41717461],
			[ 8.63429886],
			[ 8.19939795],
			[10.37567392],
			[10.68238222]])
	print("x:", x.shape)
	print("y:", y.shape)
	
	x_ = add_polynomial_features(x, 3)
	my_lr = MyLR(np.ones(4).reshape(-1,1), alpha=3.e-6, max_iter=70000)
	my_lr.fit_(x_, y)

	# Plot:
	## To get a smooth curve, we need a lot of data points
	x_steps = np.linspace(1, 11, 901)
	continuous_x = np.arange(1,10.01, 0.01).reshape(-1,1)
	x_ = add_polynomial_features(continuous_x, 3)
	y_hat = my_lr.predict_(add_polynomial_features(x_steps, 3))

	plt.scatter(x, y)
	plt.plot(continuous_x, y_hat, color='orange')
	plt.show()

def ex2():
	try:
		data = pd.read_csv("are_blue_pills_magics.csv")
	except:
		print("Invalid file error.")
		sys.exit()
	X = np.array(data[['Micrograms']])
	Y = np.array(data[['Score']])
	#print("X:", X, X.shape)
	#print("Y:", Y.shape)

	theta1 = np.random.rand(2, 1)
	theta2 = np.random.rand(3, 1)
	theta3 = np.random.rand(4, 1)
	theta4 = np.array([[-20],[ 160],[ -80],[ 10],[ -1]]).reshape(-1,1)
	theta5 = np.array([[1140],[ -1850],[ 1110],[ -305],[ 40],[ -2]]).reshape(-1,1)
	theta6 = np.array([[9110],[ -18015],[ 13400],[ -4935],[ 966],[ -96.4],[ 3.86]]).reshape(-1,1)

	model1 = MyLR(theta1, alpha = 1e-5, max_iter = 1000000)
	model2 = MyLR(theta2, alpha = 3e-5, max_iter = 1000000)
	model3 = MyLR(theta3, alpha = 1e-5, max_iter = 1000000)
	model4 = MyLR(theta4, alpha = 1e-6, max_iter = 1000000)
	model5 = MyLR(theta5, alpha = 2.5e-8, max_iter = 1000000)
	model6 = MyLR(theta6, alpha = 1e-9, max_iter = 1000000)

	X1 = add_polynomial_features(X, 1)
	X2 = add_polynomial_features(X, 2)
	X3 = add_polynomial_features(X, 3)
	X4 = add_polynomial_features(X, 4)
	X5 = add_polynomial_features(X, 5)
	X6 = add_polynomial_features(X, 6)

	print("start training model1")
	model1.fit_(X1, Y)
	print("start training model2")
	model2.fit_(X2, Y)
	print("start training model3")
	model3.fit_(X3, Y)
	print("start training model4")
	model4.fit_(X4, Y)
	print("start training model5")
	model5.fit_(X5, Y)
	print("start training model6")
	model6.fit_(X6, Y)

	# print mse and plot a bar plot
	mse_scores = []
	for i, x, model in zip(range(1, 7), 
				[X1, X2, X3, X4, X5, X6], 
				[model1, model2, model3, model4, model5, model6]):
		mse_scores.append(model.mse_(Y, model.predict_(x)))
		print(f"model{i}: mse = {mse_scores[i - 1]}")

	plt.bar(range(1, 7), mse_scores)
	plt.xlabel('Polynomial Degree')
	plt.ylabel('MSE Score')
	plt.show()

	# plot data scatter plot and model line plot 
	_, axe = plt.subplots(1, 1, figsize=(15, 8))
	x_steps = np.linspace(1, 7, 100)
	axe.scatter(X, Y, label='raw', c='black')
	axe.plot(x_steps, model1.predict_(add_polynomial_features(x_steps, 1)),
	         label='model 1')
	axe.plot(x_steps, model2.predict_(add_polynomial_features(x_steps, 2)),
	         label='model 2')
	axe.plot(x_steps, model3.predict_(add_polynomial_features(x_steps, 3)),
	         label='model 3')
	axe.plot(x_steps, model4.predict_(add_polynomial_features(x_steps, 4)),
	         label='model 4')
	axe.plot(x_steps, model5.predict_(add_polynomial_features(x_steps, 5)),
	         label='model 5')
	axe.plot(x_steps, model6.predict_(add_polynomial_features(x_steps, 6)),
	         label='model 6')
	plt.grid()
	plt.legend()
	plt.show()


if __name__ == "__main__":
	ex2()
