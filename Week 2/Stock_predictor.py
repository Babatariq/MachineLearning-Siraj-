import pandas as pd
import datetime
import pandas_datareader.data as web
from pandas import Series, DataFrame
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

import sklearn

from sklearn import preprocessing
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor

from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split


start = datetime.datetime(2015, 1, 1)
end = datetime.datetime(2019, 9 , 1)

df = web.DataReader("SHOP", "yahoo", start, end)

df.tail()

close_px = df['Adj Close']
M_avg = close_px.rolling(window = 60).mean() #moving avge windo 60 days using ajusted closing

# Plotting data
#%matplotlib inline
mpl.rc('figure', figsize=(8, 7))
mpl.__version__

# Adjusting the style of matplotlib
#style.use('ggplot')

close_px.plot(label='SHOP')
M_avg.plot(label='mavg')
plt.legend()

rets = close_px / close_px.shift(1) - 1 #the returns of change in close price day to day
rets.plot(label='return')
# Compare the same time frame of various stocks
#dfcomp = web.DataReader(['SHOP', 'GE', 'GOOG', 'IBM', 'MSFT'],'yahoo',start=start,end=end)['Adj Close']
# Here we will tune the array for clean up

dfreg = df.loc[:,["Adj Close","Volume"]]
dfreg['HL_PCT'] = (df["High"] - df["Low"]) / df["Close"] * 100.0
dfreg["PCT_change"] = (df["Close"] - df["Open"]) / df["Open"] * 100.0

# Dropp missing values
dfreg.fillna( value = -99999, inplace = True)

# Pull out 1 percent of data for testing / validating

forcast_out = int((0.01 * len(dfreg)))

# create the prelim training set and removing the 1%

forcast_col = 'Adj Close'
dfreg ['Label'] = dfreg[forcast_col].shift(-forcast_out)
X = np.array(dfreg.drop(['Label'], 1)) #removes the column name

X = preprocessing.scale(X) # scaled the Data this is a set of ADJ Close values for label these are the results or X
#used to generate the model
X_lately = X[-forcast_out:]
X = X[:-forcast_out]
#used for labels
y = np.array(dfreg['Label'])
y = y[:-forcast_out]

X_train,X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2)
len(X)


#Linear LinearRegression
clfreg = LinearRegression(n_jobs = -1) # -1 means uses all processors
clfreg.fit(X_train, y_train)

#Quaratic Regression
clfpoly2 = make_pipeline(PolynomialFeatures(2), Ridge())
clfpoly2.fit(X_train, y_train)

#Polynomial regresion of degree 3 (Cubic?)
clfpoly3 = make_pipeline(PolynomialFeatures(3), Ridge())
clfpoly3.fit(X_train, y_train)

#KNN Regression

clfknn = KNeighborsRegressor(n_neighbors = 2)
clfknn.fit(X_train, y_train)

#confidence scores
con_reg = clfreg.score(X_test, y_test)
con_poly2 = clfpoly2.score(X_test, y_test)
con_poly3 = clfpoly3.score(X_test, y_test)
con_knn = clfknn.score(X_test, y_test)

#print out the scores
print( con_reg,
con_poly2,
con_poly3,
con_knn)

forecast_set = clfreg.predict(X_lately)
dfreg['Forecast'] = np.nan

last_date = dfreg.iloc[-1].name
last_unix = last_date
next_unix = last_unix + datetime.timedelta(days=1)

for i in forecast_set:
    next_date = next_unix
    next_unix += datetime.timedelta(days=1)
    dfreg.loc[next_date] = [np.nan for _ in range(len(dfreg.columns)-1)]+[i]
dfreg['Adj Close'].tail(500).plot()
dfreg['Forecast'].tail(500).plot()
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()
