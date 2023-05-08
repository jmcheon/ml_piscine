import TinyStatistician as ts
import numpy as np

if __name__ == "__main__":
	data = [42, 7, 69, 18, 352, 3, 650, 754, 438, 2659]
	epsilon = 1e-5
	err = "Error, grade 0 :("
	tstat = ts.TinyStatistician()
	print("np.median:", np.median(data))
	print("np.percentile(data, 10):", np.percentile(data, 10))
	print("tstat.percentile(data, 10):", tstat.percentile(data, 10))
	assert abs(tstat.mean(data) - 499.2) < epsilon, err
	assert abs(tstat.median(data) - 210.5) < epsilon, err
	quartile = tstat.quartile(data)
	assert abs(quartile[0] - 18) < epsilon, err
	assert abs(quartile[1] - 650) < epsilon, err
	#assert abs(tstat.percentile(data, 10) - 3) < epsilon, err
	#assert abs(tstat.percentile(data, 28) - 18) < epsilon, err
	#assert abs(tstat.percentile(data, 83) - 754) < epsilon, err
	print("tstat.var(data):", tstat.var(data))
	print("abs(tstat.var(data) - 654661):", abs(tstat.var(data) - 654661))
	assert abs(tstat.var(data) - 654661) < epsilon, err
	assert abs(tstat.std(data) - 809.11) < epsilon, err
