import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
train = pd.read_csv('data.csv')
train.SalePrice.describe()

##Null values
nulls = pd.DataFrame(train.isnull().sum().sort_values(ascending=False)[:25])
nulls.columns = ['Null Count']
nulls.index.name = 'Feature'

##handling missing value
data = train.select_dtypes(include=[np.number]).interpolate().dropna()

#Plotting
plt.scatter(train['GarageArea'],train['SalePrice'], alpha=.75, color='b')
plt.ylabel('Sale Price')
plt.xlabel('Garage Area')
plt.title('Scatter Plot for Price VS. Garage Area')
plt.show()

#remove outliers
z = np.abs(stats.zscore(train['GarageArea']))
threshold = 2
print (np.where(z < 2))
removed = train["GarageArea"][(z<2).all(axis=1)]