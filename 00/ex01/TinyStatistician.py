import numpy as np

class TinyStatistician:
	"""
		• mean(x), median(x), quartile(x), percentile(x, p), var(x), std(x)
	"""
	def mean(self, x):
		if not isinstance(x, list) or len(x) == 0:
			return None
		result = 0.0
		for elem in x:
			result += elem
		return result / len(x)

	def median(self, x):
		if not isinstance(x, list) or len(x) == 0:
			return None
		numbers = sorted(x)
		middle = len(numbers) // 2
	#	print("==============median()===============")
	#	print("sorted list:", numbers) 
	#	print("len(lst) / 2 = ", len(numbers) / 2)
	#	print("middle index:", middle)
	#	print("=====================================\n")
		if len(numbers) % 2 == 0:
			return float(numbers[middle - 1])
		else:
			return float(numbers[middle])

	def quartile(self, x):
		if not isinstance(x, list) or len(x) == 0:
			return None
		numbers = sorted(x)
		n = len(x)
		# q1 > 1/4 * n
		q1 = self.median(numbers[:(n + 1)//2])
		# q3 > 2/4 * n
		q3 = self.median(numbers[(n + 1)//2:])
		return [q1, q3]

	def percentile(self, x, p):
		"""
		it computes the expected percentile of a given non-empty list or array x.
		The method returns the percentile as a float, otherwise None if x is an empty list or array or a non expected type object.
		The second parameter is the wished percentile. This method should not raise any Exception.
		"""
		if not isinstance(x, (np.ndarray, list)) or len(x) == 0:
			return None
		sorted_x = sorted(x)
		#print("sorted x:", sorted_x, f"p:{p}%\n")
		# Bessel's correction: len(x) - 1 instead of len(x)
		index = (len(x) - 1) * p / 100.0
		if index.is_integer():
			return sorted_x[int(index)]
		else:
			k = int(index) # the index of lower value nearest to the desired percentile
			d = index - k # decimal part of index
			#print(f"index:{index}, k(int part):{k}, d(decimal part):{d}\n")
			#print(f"sorted_x[{k}]: {sorted_x[k]}")
			#print(f"sorted_x[{k + 1}]: {sorted_x[k + 1]}")

			#print(f"sorted_x[{k}] * {1 - d}: {sorted_x[k] * (1 - d)}")
			#print(f"sorted_x[{k + 1}] * {d}: {sorted_x[k + 1] * d}\n")

		#	print(f"\nsorted_x[k] * (1 - d) + sorted_x[k + 1] * d")
		#	print(f"sorted_x[{k}] * {1 - d} + sorted_x[{k + 1}] * {d}")
		#	print(f"{sorted_x[k]} * {1 - d} + {sorted_x[k + 1]} * {d}")
		#	print(f"{sorted_x[k] * (1 - d)} + {sorted_x[k + 1] * d}")
			return sorted_x[k] * (1 - d) + sorted_x[k + 1] * d

	def var(self, x):
		if not isinstance(x, list) or len(x) == 0:
			return None
		mean = self.mean(x)
		suqared_diff_sum = 0.0
		for num in x:
			suqared_diff_sum += (num - mean) ** 2
		return suqared_diff_sum / (len(x) - 1)

	def std(self, x):
		if not isinstance(x, list) or len(x) == 0:
			return None
		return self.var(x) ** 0.5

def ex1(t):
	a = [1, 42, 300, 10, 59]
	#a = [] 
	print(t.mean(a)) # 82.4
	print(t.median(a)) # 42
	print(t.quartiles(a)) # 10 59
	print(t.var(a)) # 12279.439999999999
	print(t.std(a)) # 110.81263465868862

def ex2(t):
	lst = [14, 17, 10, 14, 18, 20, 13]
	print(t.mean(lst)) # 
	print(t.median(lst)) # 14
	print(t.quartiles(lst)) # 13, 18
	print(t.var(lst)) # 
	print(t.std(lst)) # 

def ex3(t):
	lst2 = [177, 180, 175, 182, 190, 169, 185, 191, 193]
	print(t.mean(lst2)) # 
	print(t.median(lst2)) # 182
	print(t.quartiles(lst2)) # 177 190
	print(t.var(lst2)) # 
	print(t.std(lst2)) # 

def ex4():
	a = [1, 42, 300, 10, 59]
	print(TinyStatistician().mean(a)) # Output: 82.4
	print(TinyStatistician().median(a)) # Output: 42.0
	print(TinyStatistician().quartile(a)) # Output: [10.0, 59.0]
	print(TinyStatistician().var(a)) # Output: 15349.3
	print(TinyStatistician().std(a)) # Output: 123.89229193133849

def ex5():
	a = [1, 42, 300, 10, 59]
	print(TinyStatistician().percentile(a, 10)) # Output: 4.6
	print(TinyStatistician().percentile(a, 15)) # Output: 6.4
	print(TinyStatistician().percentile(a, 20)) # Output: 8.2
	print(TinyStatistician().percentile(a, 80)) # Output: 8.2

if __name__ == "__main__":
	t = TinyStatistician()
	#ex1(t)
	ex4()
