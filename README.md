# Macchine-learning-Time-Series-Forecasting

2 Time Series and GD in Python and NumPy

From a Jupyter notebook in Python, extract one single time series of your choice from:
the Quandl API, call it xt
the CryptoCompare API, call it yt
Depending on availability, choose your own frequency and time period, making sure that you can analyse both time series by treating them each as a one dimensional array of the same length. Use the numpy library to

find OLS estimates of α and β in the following specification, where et is an assumed white noise error:
yt = α + βxt + et
Analytically from standard OLS formulae
By trial and error Machine Learning with a Gradient Descent (GD) Algorithm

4 Time Series Forecasting

In this section you are set an open ended mission of using everything you have learned to create a Python notebook that produces a time series forecast of a financial time series of your choice. Pick one single financial time series that you are interested in (eg the Bitcoin SV price in Sterling), and conduct your analysis in stages: In the first stage, you must describe why you have chosen the series, and what factors you think affect it. Where relevant, describe any general trends, specific events, and what variables you think may drive it. Distinguish between variables that you should be able to obtain, versus those you cannot. In the second stage, conduct your analysis. There are many potential difficulties to overcome, including extracting the relevant time series (you could use the quandl or cryptocompare API calls already used, or any other).
Your analysis should reduce a larger model with many variables (more than 10) down to a smaller model by applying a Machine Learning Technique like L1/L2 Regularisation. Scikit learn has examples of implementing the combination of L1 and L2 called elastic nets. This is the recommended approach, but you can also choose other machine learning approaches as long as they are properly documented. Include an assessment of how good your implementation is.
