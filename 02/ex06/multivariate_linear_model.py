import numpy as np
import pandas as pd
import sys, os
import matplotlib.pyplot as plt
path = os.path.join(os.path.dirname(__file__), '..', 'ex05')
sys.path.insert(1, path)
from mylinearregression import MyLinearRegression as MyLR

def plot_regression_model(X, Y, y_pred, x_label, colors, loc):
	plt.scatter(X, Y, label="Sell price", color=colors[0])
	plt.scatter(X, y_pred, label="Predicted sell price", color=colors[1], s=8)
	plt.xlabel(f"{x_label}")
	plt.ylabel("y: sell price(in keuros)")
	plt.legend(loc=loc)
	plt.grid()
	plt.show()

def univariate_model(x_variable, x_label, colors, loc, thetas):
	X = np.array(data[[x_variable]])
	Y = np.array(data[['Sell_price']])
	myLR = MyLR(thetas, alpha = 2.5e-5, max_iter = 200000)
	myLR.fit_(X[:,0].reshape(-1,1), Y)
	y_pred = myLR.predict_(X[:,0].reshape(-1,1))

	plot_regression_model(X, Y, y_pred, x_label, colors, loc)
	print("univariate model feature x:", x_variable)
	print("thetas(oringal):", thetas)
	print("thetas(new):", myLR.thetas)
	print("mse:", myLR.mse_(y_pred,Y), "\n") 

def part_one():
	univariate_model('Age', 
			"$x_{1}$: age (in years)", 
			('darkblue', 'cornflowerblue'),
			'lower left',
			np.array([[700.0], [-1.0]]))
	univariate_model('Thrust_power', 
			"$x_{2}$: thrust power(in 10Km/s)", 
			 ('green', 'lime'),
			'upper left',
			np.array([[0.0], [-1.0]]))
	univariate_model('Terameters', 
			"x_{3}$: distance totalizer value of spcaecraft (in Tmeters)", 
			('purple', 'violet'),
			'upper right',
			np.array([[700.0], [-1.0]]))

def multivariate_model():
	X = np.array(data[['Age','Thrust_power','Terameters']])
	Y = np.array(data[['Sell_price']])
	thetas = np.array([1.0, 1.0, 1.0, 1.0])
	my_lreg = MyLR(thetas, alpha = 1e-5, max_iter = 500000)

	# Example 1:
	my_lreg.fit_(X,Y)
	y_pred = my_lreg.predict_(X)

	plot_regression_model(X[:,0], Y, y_pred, 
				"$x_{1}$: age (in years)",
				('darkblue', 'cornflowerblue'),
				'lower right')
	plot_regression_model(X[:,1], Y, y_pred, 
				"$x_{1}$: age (in years)",
				('green', 'lime'),
				'upper left')
	plot_regression_model(X[:,2], Y, y_pred, 
				"$x_{1}$: age (in years)",
				('purple', 'violet'),
				'upper right')
	print("multivariate model:")
	print("thetas(oringal):", thetas)
	print("thetas(new):", my_lreg.thetas)
	print("mse:", my_lreg.mse_(y_pred,Y), "\n") 

def part_two():
	multivariate_model()


if __name__ == "__main__":
	try:
		data = pd.read_csv("spacecraft_data.csv")
	except:
		print("Invalid file error.")
		sys.exit()
	part_one()
	part_two()
