

import pandas as pd
from numpy.distutils.system_info import p

data = pd.read_csv(
    'https://raw.githubusercontent.com/dataprofessor/data/master/delaney_solubility_with_descriptors.csv')

# DATA PREPARATION
""" Data separation as X and Y"""

y = data['logS']
print(y)

x = data.drop('logS', axis=1)
print(x)

"""Data splitting"""

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=100)
print(x_train)

""" Model Building"""
# Linear Regression #

from sklearn.linear_model import LinearRegression

lr = LinearRegression()
lr.fit(x_train, y_train)

LinearRegression()

"""Applying the model to make a prediction """

y_lr_train_pred = lr.predict(x_train)
y_lr_test_pred = lr.predict(y_train)

print(y_lr_train_pred, y_lr_test_pred)

"""Evaluate model preformance """

from sklearn.metrics import mean_squared_error, r2_score

lr_train_mae = mean_squared_error(y_train, y_lr_train_pred)
lr_train_r2 = r2_score(y_train, y_lr_train_pred)

lr_test_mae = mean_squared_error(y_train, y_lr_train_pred)
lr_test_r2 = r2_score(y_train, y_lr_test_pred)

print('LR MSE (Train):', lr_train_mae)
print('LR R2 (Train):', lr_train_r2)
print('LR MSE (Train):', lr_test_mae)
print('LR R2 (Train):', lr_test_r2)

lr_results = pd.DataFrame(['Linear regression ', lr_train_mae, lr_train_r2, lr_test_mae, lr_test_r2]).transpose()
lr_results.colums = ['Method', 'Training MSE', 'Training R2', 'Test MSE', 'Test R2']
print(lr_results)

"""Data Visualization of prediction results """
import matplotlib.pyplot as plt

plt.figure(figsize=(5, 5))
plt.scatter(x=y_train, y=y_lr_train_pred, alpha=0.3)

plt.plot(y_train, p(y_train), '#F8766D')
plt.ylabel('Predict LogS')
plt.xlabel('Experimental LogS')
