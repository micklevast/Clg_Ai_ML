# Here is a more elegant and scalable solution, imo. It'll work for any nxn matrix and you may find use for the other methods.
# Note that getMatrixInverse(m) takes in an array of arrays as input. Please feel free to ask any questions
import numpy as np
import matplotlib.pyplot as plt
import sys


from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures


def transposeMatrix(m):
    return map(list, zip(*m))


def getMatrixMinor(m, i, j):
    return [row[:j] + row[j+1:] for row in (m[:i]+m[i+1:])]


def getMatrixDeternminant(m):
    # base case for 2x2 matrix
    if len(m) == 2:
        return m[0][0]*m[1][1]-m[0][1]*m[1][0]

    determinant = 0
    for c in range(len(m)):
        determinant += ((-1)**c)*m[0][c] * \
            getMatrixDeternminant(getMatrixMinor(m, 0, c))
    return determinant


def getMatrixInverse(m):
    determinant = getMatrixDeternminant(m)
    # special case for 2x2 matrix:
    if len(m) == 2:
        return [[m[1][1]/determinant, -1*m[0][1]/determinant],
                [-1*m[1][0]/determinant, m[0][0]/determinant]]

    # find matrix of cofactors
    cofactors = []
    for r in range(len(m)):
        cofactorRow = []
        for c in range(len(m)):
            minor = getMatrixMinor(m, r, c)
            cofactorRow.append(((-1)**(r+c)) * getMatrixDeternminant(minor))
        cofactors.append(cofactorRow)
    cofactors = transposeMatrix(cofactors)

    # len(list(L)
    for r in range(len(list(cofactors))):
        for c in range(len(list(cofactors))):
            cofactors[r][c] = cofactors[r][c]/determinant
    return cofactors


def zeros_matrix(rows, cols):
    A = []
    for i in range(rows):
        A.append([])
        for j in range(cols):
            A[-1].append(0.0)

    return A


def matrix_multiply(A, B):
    # rowsA = len(list(A))
    # colsA = len(list(A))

    # rowsB = len(list(B))
    # colsB = len(list(B))
    rowsA = 3
    colsA = 3
    rowsB = 3
    colsB = 1

    if colsA != rowsB:
        print('Number of A columns must equal number of B rows.')
        sys.exit()

    C = zeros_matrix(rowsA, colsB)

    for i in range(rowsA):
        for j in range(colsB):
            total = 0
            for ii in range(colsA):
                total += A[i][ii] * B[ii][j]
            C[i][j] = total

    return C


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

Inv_Matrix_X = getMatrixInverse(Matrix_X)

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

Ans_as_all_coefficient = matrix_multiply(Inv_Matrix_X, Matrix_Y)

for i in range(len(Ans_as_all_coefficient)):
    for j in range(len(Ans_as_all_coefficient[0])):
        print(Ans_as_all_coefficient[i][j])
