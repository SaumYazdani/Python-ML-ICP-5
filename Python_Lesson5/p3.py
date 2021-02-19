import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
train = pd.read_csv('restaurantdata')
# map train values to their correlation values
correlation = train.corr().abs()
# set equality (self correlation) as zero
correlation[correlation == 1] = 0
#find the max correlation
# and sort in ascending order
sortedcorr = correlation.max().sort_values(ascending=False)
# display the highly correlated values
print(sortedcorr[sortedcorr > 0.8])
##handling missing value
data = train.select_dtypes(include=[np.number]).interpolate().dropna()
##Build a linear model, dropping all values that aren't the top 5 correlated attributes
y = np.log(train.revenue)
X = data.drop(['Id','P1','P2','P3','P4','P5','P6','P7','P8','P10','P11','P13','P14','P15',
               'P17','P18','P19','P20','P21','P22','P23','P24','P25','P26','P27','P28','P29','P30','P31','P32','P33','P35', 'P37'], axis=1)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=.33)
from sklearn import linear_model
lr = linear_model.LinearRegression()
model = lr.fit(X_train, y_train)
##Evaluate the performance and visualize results
print ("R^2 is: \n", model.score(X_test, y_test))
predictions = model.predict(X_test)
from sklearn.metrics import mean_squared_error
print ('RMSE is: \n', mean_squared_error(y_test, predictions))
##visualize
actual_values = y_test
plt.scatter(predictions, actual_values, alpha=.75, color='b') #alpha helps to show overlapping data
plt.xlabel('Predicted Revenue')
plt.ylabel('Actual Revenue')
plt.title('Linear Regression Model')
plt.show()
