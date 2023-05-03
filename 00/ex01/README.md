



Create a class named **TinyStatistician** that implements the following methods: 

 - mean(x): computes the mean of a given non-empty list or array x, using a for-loop. The method returns the mean as a float, otherwise None if x is an empty list or array, or a non expected type object. This method should not raise any Exception. Given a vector x of dimension m * 1, the mathematical formula of its mean is:<br/> $$\bar{x} = \frac{ 1}{m}\sum_{i=1}^{m} x_i$$ 


 - median(x): computes the median, which is also the 50th percentile, of a given nonempty list or array x . The method returns the median as a float, otherwise None if x is an empty list or array or a non expected type object. This method should not raise any Exception.
 
 - quartile(x): computes the 1st and 3rd quartiles, also called the 25th percentile and the 75th percentile, of a given non-empty list or array x. The method returns the quartiles as a list of 2 floats, otherwise None if x is an empty list or array or a non expected type object. This method should not raise any Exception.
 
 - percentile(x, p): computes the expected percentile of a given non-empty list or array x. The method returns the percentile as a float, otherwise None if x is an empty list or array or a non expected type object. The second parameter is the wished percentile. This method should not raise any Exception.

 - var(x): computes the sample variance of a given non-empty list or array x. The method returns the sample variance as a float, otherwise None if x is an empty list or array or a non expected type object. This method should not raise any Exception.<br/> Given a vector x of dimension m * 1 representing the a sample of a data population, the mathematical formula of its variance is:<br/> $$s^2 = \frac{1} {m-1}\sum_{i=1}^{m} (x_i - \bar{x})^2  = \frac{1} {m-1}\sum_{i=1}^{m} [x_i - (\frac{1} {m}\sum_{j=1}^{m} x_j)]^2$$  

 - std(x): computes the sample standard deviation of a given non-empty list or array x. The method returns the sample standard deviation as a float, otherwise None if x is an empty list or array or a non expected type object. This method should not raise any Exception. Given a vector x of dimension m * 1, the mathematical formula of the sample standard deviation is:<br/> $$s = \sqrt{\frac{1} {m-1}\sum_{i=1}^{m} (x_i - \bar{x})^2}  = \sqrt{\frac{1} {m-1}\sum_{i=1}^{m} [x_i - (\frac{1} {m}\sum_{j=1}^{m} x_j)]^2}$$
