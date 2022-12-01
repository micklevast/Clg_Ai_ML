# Here is a more elegant and scalable solution, imo. It'll work for any nxn matrix and you may find use for the other methods.
# Note that getMatrixInverse(m) takes in an array of arrays as input. Please feel free to ask any questions
import numpy as np
import matplotlib.pyplot as plt
import sys


from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures


def zeros_matrix(rows, cols):
    A = []
    for i in range(rows):
        A.append([])
        for j in range(cols):
            A[-1].append(0.0)

    return A
# # Taking a 3 * 3 matrix
# A = np.array([[6, 1, 1],
#               [4, -2, 5],
#               [2, 8, 7]])


def mulAB(A, B, result):
    # iterating by row of A
    for i in range(len(A)):
        # print("i", A[i])

        # iterating by column by B
        for j in range(len(B[0])):
            # print("j", B[j])

            # iterating by rows of B
            for k in range(len(B)):
                result[i][j] += A[i][k] * B[k][j]

    return result


x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
# x = x.reshape(-1, 1)
# y = np.array((1, 4, 9, 15, 23, 35, 46, 62, 75, 97), ndmin=2)
y = np.array([1, 4, 9, 15, 23, 35, 46, 62, 75, 97])
# y = y.reshape(10)


Matrix_X = zeros_matrix(3, 3)
row = 3
col = 3
for i in range(row):
    for j in range(col):
        Matrix_X[i][j] = 0  # firsly initilised as zero
        if i == 0 and j == 0:
            Matrix_X[i][j] = len(x)
        else:
            power_of_x = i+j
            summation_x_to_power_of_i_plus_j = 0
            for val in x:
                summation_x_to_power_of_i_plus_j = summation_x_to_power_of_i_plus_j + \
                    pow(val, power_of_x)
            Matrix_X[i][j] = summation_x_to_power_of_i_plus_j

# Calculating the inverse of the matrix
# print(np.linalg.inv(A))
Inv_Matrix_X = np.linalg.inv(Matrix_X)

# for i in range(row):
#     for j in range(col):
#         print(Matrix_X[row][col])

Matrix_Y = zeros_matrix(3, 1)


def summation_y(y):
    sum = 0
    for val in y:
        sum += val
    return sum


def summation_xy(x, y):
    sum = 0
    for i in range(len(x)):
        sum = sum+(y[i]*x[i])
    return sum


def summation_x_square_y(x, y):
    sum = 0
    for i in range(len(x)):
        sum = sum+(y[i]*(pow(x[i], 2)))
    return sum


Matrix_Y[0][0] = summation_y(y)
Matrix_Y[1][0] = summation_xy(x, y)
Matrix_Y[2][0] = summation_x_square_y(x, y)

# --------------------------------
result = [[0],
          [0],
          [0]]

Ans_as_all_coefficient = mulAB(Inv_Matrix_X, Matrix_Y, result)
# print(Matrix_X)
# print(Inv_Matrix_X)
# print(Matrix_Y)
print(Ans_as_all_coefficient)

# for i in range(len(Ans_as_all_coefficient)):
#     for j in range(len(Ans_as_all_coefficient[0])):
#         print(Ans_as_all_coefficient[i][j])
print("best fit line will be:[a0+a1x+a2x^2]:", str(Ans_as_all_coefficient[0][0]) +
      " + " + str(Ans_as_all_coefficient[1][0])+" *x  + "+str(Ans_as_all_coefficient[2][0])+" *x^2")
