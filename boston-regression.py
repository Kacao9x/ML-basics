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
plt.scatter( bos['RM'], bos['PRICE'])
plt.xlabel("Avg of Room per Dwelling RM")
plt.ylabel("")
plt.title("Relationship btw House Price and RM")
plt.show()