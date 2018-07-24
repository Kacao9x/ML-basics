import pandas as pd
import numpy as np
import sklearn
import scipy.stats as stats
import matplotlib.pyplot as plt

# Retrieve data as a dictionary object from a given database
from sklearn.datasets import load_boston
boston = load_boston()
print (boston.data.shape)
print (boston.keys())
print (boston['feature_names'])

# load the dataset into Dataframe with a default column names
bs = pd.DataFrame( boston.data )
bs.columns = boston['feature_names']
print (bs.head())

# first 5 value of the target: housing price
print (boston['target'][:5])