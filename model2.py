# -*- coding: utf-8 -*-
"""
Created on Thu Jan  4 21:53:38 2018

@author: Utkarsh
"""


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
deg=1; min = float('Inf'); apt_deg = 1
while deg<=10:
    poly_reg = PolynomialFeatures(degree = deg)
    X_poly_temp = poly_reg.fit_transform(X)
    poly_reg.fit(X_poly_temp, y)
    lin_reg_2 = LinearRegression()
    lin_reg_2.fit(X_poly_temp, y)
    y_pred = lin_reg_2.predict(X_poly_temp)
    mse = 0
    for i in range(10):
        mse = mse + ((y_pred[i]-y[i])**2)
    mse/=20
    if mse < min:
        min = mse
        apt_deg = deg
    deg+=1

poly_reg = PolynomialFeatures(degree = apt_deg)
X_poly = poly_reg.fit_transform(X)
poly_reg.fit(X_poly, y)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, y)

plt.scatter(X, y, color = 'red')
plt.plot(X, lin_reg_2.predict(poly_reg.fit_transform(X)), color = 'blue')
plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()