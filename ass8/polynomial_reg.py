# https://www.javatpoint.com/machine-learning-polynomial-regression

# Steps for Polynomial Regression:
# The main steps involved in Polynomial Regression are given below:

# Data Pre-processing
# Build a Linear Regression model and fit it to the dataset
# Build a Polynomial Regression model and fit it to the dataset
# Visualize the result for Linear Regression and Polynomial Regression model.
# Predicting the output.

# importing libraries
import numpy as nm
import matplotlib.pyplot as mtp
import pandas as pd

# importing datasets
data_set = pd.read_csv('Position_Salaries.csv')

# Extracting Independent and dependent Variable
x = data_set.iloc[:, 1:2].values
y = data_set.iloc[:, 2].values
