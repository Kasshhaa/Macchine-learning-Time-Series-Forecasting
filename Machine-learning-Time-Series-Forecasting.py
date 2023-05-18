import pandas as pd
import quandl
import requests
import json
import numpy as np
import matplotlib.pyplot as plt
import requests
import datetime
import cryptocompare
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Lasso
import warnings
warnings.filterwarnings("ignore")

#2. Time Series and GD in Python and NumPy

# Getting quandl API
key_1 = quandl.ApiConfig.api_key = "JepaoR-rKUDTh3xZw2Xr"

#Generating time series of Silver
silver = quandl.get("LBMA/SILVER", start_date="2020-06-9", end_date="2020-07-04")
xt_data = pd.DataFrame.from_dict(silver)
xt= xt_data["GBP"]

# Getting cryptoCompare API
cryptocompare_API_key = "bbff4feae034140397b945771b365bcf41b3bf6f3e4e0df64bb69670b498638a"
cryptocompare.cryptocompare._set_api_key_parameter(cryptocompare_API_key)

# Generating time series of btc price and setting date time index
btc= cryptocompare.get_historical_price_day('BTC', 'GBP', limit=24, exchange='CCCAGG', toTs=datetime.datetime(2020,7,4))
btc_data = pd.DataFrame.from_dict(btc)
btc_data.set_index("time", inplace=True)
btc_data.index = pd.to_datetime(btc_data.index, unit='s')
btc_data['datetimes'] = btc_data.index
btc_data['datetimes'] = btc_data['datetimes'].dt.strftime(
    '%Y-%m-%d')
yt= btc_data["close"]

# Ensuring that both Btc and Silver are the same length
df ={}
df['Silver'] = xt_data["GBP"]
df["Btc"] = btc_data["close"]
df = pd.DataFrame(df).fillna(method ="ffill")

# Creating a one dimensional array of both Btc and Silver
x= np.array(df["Silver"])
y= np.array(df["Btc"])

#Finding OLS estimates from standard OLS formulae

# Use numpy to estimate parameters α and β
beta = ((np.multiply(y, x)).mean()-np.mean(x)*np.mean(y))/((np.multiply(x, x)).mean()-np.mean(x)*np.mean(x))
alpha =np.mean(y)-beta*np.mean(x)

print("Estimated alpha (intercept):", alpha)
print("Estimated beta (slope):", beta)


y_hat0 = alpha + np.multiply(beta,x)
L = np.sum(np.multiply(y - y_hat0,y - y_hat0))

#Finding OLS estimates from Machine Learning with a Gradient Descent

alpha = 10310

y_hat = alpha + np.multiply(beta,x)


iterations = 500
learningrate = 0.01
def  cal_cost(alpha,beta,x,y):
        n = len(y)
        predictions = alpha + np.multiply(beta,x)
        cost = (1/2*n) * np.sum(np.square(predictions-y))
        return cost

cal_cost(alpha,beta,x,y)

cal_cost(10310,beta,x,y)

iterations = 21
alpha = 10291

loss_history = np.zeros((iterations,1))
alpha_history = np.zeros((iterations,1))

for i in range(iterations):
    alpha = alpha +1

    loss_history[i] = cal_cost(alpha,beta,x,y)
    alpha_history[i] = alpha

plt.plot(alpha_history,loss_history,'go')

iterations = 1000
alpha = 10291
n = len(y)

loss_history = np.zeros((iterations,1))
alpha_history = np.zeros((iterations,1))

for i in range(iterations):


        prediction = alpha + np.multiply(beta,x)
        residuals = y-prediction
        residuals_sum = sum(residuals)
        gradient = -(1/n)*0.1*residuals_sum

        alpha = alpha - gradient

        loss_history[i] = cal_cost(alpha,beta,x,y)
        alpha_history[i] = alpha

plt.plot(alpha_history,loss_history,'go')

#4. Time Series Forecasting

btc_coins = pd.DataFrame()
my_crypo_key = " "

URL_A = "https://min-api.cryptocompare.com/data/v2/histoday?fsym=BTC&tsym=USD&limit=10 &api_key= JepaoR-rKUDTh3xZw2Xr"
URL = URL_A + my_crypo_key
data = requests.get(URL)
json_data = data.json()
table = pandas.json_normalize(datah,'Data').set_index('time')
table.index = pandas.to_datetime(table.index, unit='s')
btc_coins = pd.concat ([btc_coins, table.high], axis=1)

btc_coins
coins_lagged = coins.copy ()
trailing_window_size = 1
for window in range(1, trailing_window_size +1):
    shifted = coins.shift(window)
    shifted.columns = [x + " lag" + str(window) for x in coins. columns]
    coins_lagged = pd. concat((coins_lagged, shifted), axis=1)
coins_lagged = coins_lagged.dropna()


coins_lagged.to_csv('crypto_prices_data.csv', index_label='time')

df = pd.read_csv('crypto_prices_data.csv', index_col='time')

#Making sure the index is Pandas datetime
#Using normalize to mke it daily analysis
df.index = pd.to_datetime(df.index)
df.index = df.index.normalize ()

#Importing the sklearn standard scalar Library
from sklearn.preprocessing import StandardScaler


#Scaling data
sc_x = StandardScaler ()
df_scaled = pd.DataFrame (sc_x.fit_transform(df), index=df.index, columns=df.columns)

split = int(int(lim)/2)
Ytrain = pd.DataFrame(data_scaled[:split]["BTC"])
Ytest = pd.DataFrame(data_scaled[split:]["BTC"])

x = data_scaled[["BTC_lag_1", "BTC_lag_2", "Trans_Vol", "Difficulty", "Hash_Rate",
                 "BTC/USD", "CPI", "No_Trans", "Cost/Trans", "ETH/BTC", "ETH/USD"]]

#Applying regularisation with elasticnet
from sklearn.linear_model import ElasticNet
en = ElasticNet(alpha=0.1, l1_ratio=0.5)
en.fit(x[:split], Ytrain)

coef = list(en.coef_)
count = 0
for i in X.columns:
    print(i, ':' , coef [count])
    count = count + 1

sample_size = len(df_scaled)

# Splitting data in two parts.
split = int(int(sample_size)/2)

# Setting the y variable of interest to forecast BTC and splitting it into training and test sets
Ytrain = pd.DataFrame(data_scaled[:split]["BTC"])
Ytest = pd.DataFrame(data_scaled[split:]["BTC"])

x=data_scaled[["BTC_lag_1", "BTC_lag_2", "Trans_Vol", "Difficulty", "Hash_Rate",
                 "BTC/USD", "CPI", "No_Trans", "Cost/Trans", "ETH/BTC", "ETH/USD"]]

en = ElasticNet(alpha=0.1, l1_ratio=0.5)
en.fit(x[:split], Ytrain)

# Converting the en.coef to a list
coef = list(en.coef_)
count = 0
for i in X.columns:
    print(i, ":", coef[count])
    count = count + 1

#Applying regularisation with elasticnet
from sklearn.linear_model import ElasticNet
​
en = ElasticNet(alpha=0.1, l1_ratio=0.5)
en.fit(Xtrain, Ytrain)

coef = list(en.coef_)

count = 0
for i in X.columns:
    print(i, ":", coef[count])
    count = count + 1

#Testing prediction
ypred = en.predict(Xtest)
print("Predicted y values:", ypred)

#The Mean Absolute Error (MSE) and Rsquared (R^2) metrics is used to measure the performance

import numpy as np
from sklearn.metrics import confusion_matrix

y_true= np.array(Ytest["BTC"])
y_predict = np.array(Ytrain["BTC"])
