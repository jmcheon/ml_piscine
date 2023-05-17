import numpy as np
import pandas as pd
import sys
import matplotlib.pyplot as plt
import matplotlib.pylab as pl
from sklearn.metrics import mean_squared_error
from my_linear_regression import MyLinearRegression as MyLR

def hypothesis():
	plt.scatter(Xpill, Yscore, label="$S_{true(pills)}$", color="cornflowerblue")
	plt.plot(Xpill, linear_model.predict_(Xpill), label="$S_{predict(pills)}$", color="limegreen", linestyle="dashed")
	plt.xlabel("Quantity of blue pill (in micrograms)")
	plt.ylabel("Space driving score")
	plt.legend(loc='upper right')
	plt.grid()
	plt.show()

def show_losses():
	n = 6
	theta0 = np.linspace(70, 100, n)
	theta1 = np.linspace(-14, -4, 100)
	lst_thetas = []
	for t0 in theta0:
		sublist = [[t0, t1] for t1 in theta1]
		lst_thetas.append([sublist])
	
	colors = pl.cm.Greys(np.linspace(1, 0, n + 1))
	fig, axe = plt.subplots(1, 1)
	for thetas, color in zip(lst_thetas, colors):
		lst_loss = []
		for theta in thetas[0]:
			linear_model = MyLR(np.array(theta))
			y_hat = linear_model.predict_(Xpill)
			lst_loss.append(linear_model.loss_(Yscore, y_hat))
		axe.plot(theta1, np.array(lst_loss), label=r"J($\theta_0$ = " + f"{theta[0]}, " + r"$\theta_1$)", lw=2.5, c=color)
	plt.grid()
	plt.legend(loc="lower right")
	plt.xlabel(r"$\theta_1$")
	plt.ylabel(r"cost function J($\theta_0 , \theta_1$)")
	axe.set_ylim([10, 150])
	axe.set_xlim([-14, -4])
	plt.show()

def print_mse():
	linear_model1 = MyLR(np.array([[89.0], [-8]]))
	linear_model2 = MyLR(np.array([[89.0], [-6]]))

	Y_model1 = linear_model1.predict_(Xpill)
	Y_model2 = linear_model2.predict_(Xpill)

	print(MyLR.mse_(Yscore, Y_model1)) # 57.60304285714282
	print(mean_squared_error(Yscore, Y_model1)) # 57.603042857142825
	print(MyLR.mse_(Yscore, Y_model2)) # 232.16344285714285
	print(mean_squared_error(Yscore, Y_model2)) # 232.16344285714285

if __name__ == "__main__":
	try:
		data = pd.read_csv("are_blue_pills_magics.csv")
	except:
		print("Invalid file error.")
		sys.exit()

	Xpill = np.array(data['Micrograms']).reshape(-1,1)
	Yscore = np.array(data['Score']).reshape(-1,1)
	thetas = np.random.rand(2, 1)
	linear_model = MyLR(thetas, alpha=5e-2, max_iter=1000)
	linear_model.fit_(Xpill, Yscore)

	hypothesis()
	show_losses()
	print_mse()
