import operator

import numpy as np
import matplotlib.pyplot as plt


from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures


x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
x = x.reshape(-1, 1)
# y = np.array((1, 4, 9, 15, 23, 35, 46, 62, 75, 97), ndmin=2)
y = np.array([1, 4, 9, 15, 23, 35, 46, 62, 75, 97])
y = y.reshape(10)

# firstly checked point scatteredness

# plt.scatter(x, y, s=10)
# plt.show()
# --------------------------


# Let’s apply a linear regression model to this dataset.
# model = LinearRegression()
# model.fit(x, y)
# y_pred = model.predict(x)

# plt.scatter(x, y, s=10)
# plt.plot(x, y_pred, color='r')
# plt.show()

# ----------------------------

# We can see that the straight line is unable to capture the patterns in the data. This is an example of under-fitting.
# Computing the RMSE and R²-score of the linear line gives:

# To overcome under-fitting, we need to increase the complexity of the model.
# To generate a higher order equation we can add powers of the original features as new features. The linear model,

# can be transformed to-->
# This is still considered to be linear model as the coefficients/weights associated with the features are still linear.
# x² is only a feature. However the curve that we are fitting is quadratic in nature.

# -----------To convert the original features into their higher order terms we will use the PolynomialFeatures class provided by scikit-learn. Next, we train the model using Linear Regression.--

polynomial_features = PolynomialFeatures(degree=2)
x_poly = polynomial_features.fit_transform(x)

model = LinearRegression()
model.fit(x_poly, y)
y_poly_pred = model.predict(x_poly)

rmse = np.sqrt(mean_squared_error(y, y_poly_pred))
r2 = r2_score(y, y_poly_pred)
print("rmse:", rmse)
print("r2 score:", r2)

plt.scatter(x, y, s=10)
# sort the values of x before line plot
sort_axis = operator.itemgetter(0)
sorted_zip = sorted(zip(x, y_poly_pred), key=sort_axis)
x, y_poly_pred = zip(*sorted_zip)
plt.plot(x, y_poly_pred, color='m')
plt.show()
