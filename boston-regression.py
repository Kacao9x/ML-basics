import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import scipy.stats as stats
import matplotlib.pyplot as plt
# Retrieve data as a dictionary object from a given database
from sklearn.datasets import load_boston
boston = load_boston()
print (boston.data.shape)
print (boston.keys())
print (boston['feature_names'])

# load the dataset into Dataframe with a default column names
bos = pd.DataFrame( boston.data )
bos.columns = boston['feature_names']
print (bos.head())

# first 5 value of the target: housing price
print (boston['target'][:5])
bos['PRICE'] = boston['target']

# keep the dataframe but 'PRICE' column as X parameter
X = bos.drop( 'PRICE', axis=1 )
lm = LinearRegression()
lm.fit( X, bos['PRICE'])

print ("Estimated intercepted: ", lm.intercept_)
print ("Estimated numbers of coefficient: ", len(lm.coef_))

# output = pd.DataFrame( zip(X.columns, lm.intercept_),
#                        columns=['features', 'estimated coeff'] )
# print (output)

#plot the data
# plt.scatter( bos['RM'], bos['PRICE'])
# plt.xlabel("Avg of Room per Dwelling RM")
# plt.ylabel("House Price")
# plt.title("Relationship btw House Price and RM")
# plt.show()

# Predict the house prices based on the dataset
# print (lm.predict( X )[ : 5])
plt.scatter(bos['PRICE'], lm.predict( X ))
plt.xlabel(" True house price ")
plt.ylabel(" Predicted prices ")
plt.title("Price Y and X parameters")
plt.show()

# the error increases if we take linear reg for one feature
meanFull = np.mean( (bos['PRICE'] - lm.predict(X)) **2)
print (meanFull)

lm1 = LinearRegression()
lm.fit( X[['AGE']], bos['PRICE'])
meanFull_1 = np.mean( (bos['PRICE'] - lm.predict(X[['AGE']])) **2)
print (meanFull_1)

# Traing the dataset
X_train = X[ : -50 ]
X_test = X[ -50 : ]
Y_train = bos['PRICE'][ : -50 ]
Y_test = bos['PRICE'][ -50 : ]

lm = LinearRegression()
lm.fit( X_train, Y_train )
pred_train = lm.predict( X_train )
pred_test  = lm.predict( X_test )

print ("Fit a model X_train, and calculate MSE with Y_train:",
       np.mean((Y_train - lm.predict(X_train)) ** 2))
print ("Fit a model X_train, and calculate MSE with X_test, Y_test:",
       np.mean((Y_test - lm.predict(X_test)) ** 2))