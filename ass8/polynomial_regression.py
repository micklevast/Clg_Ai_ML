import numpy as np
import matplotlib.pyplot as plt

A = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
b = np.array((1, 4, 9, 15, 23, 35, 46, 62, 75, 97), ndmin=2).T
b = b.reshape(10)

xfeature = A
squaredfeature = A ** 2
ones = np.ones(10)

b = np.array((1, 4, 9, 15, 23, 35, 46, 62, 75, 97), ndmin=2).T
b = b.reshape(10)


order = 2

features = np.concatenate((np.vstack(ones), np.vstack(
    xfeature), np.vstack(squaredfeature)), axis=1)
print(features)
xstar = np.matmul(np.matmul(np.linalg.inv(
    np.matmul(features.T, features)), features.T), b)
print(xstar)
plt.scatter(A, b, c='red')
u = np.linspace(0, 3, 20)
plt.plot(u, u**2*xstar[2] + u*xstar[1] + xstar[0], 'b.')
p2 = np.polyfit(A, b, 2)
plt.plot(u, np.polyval(p2, u), 'r--')
plt.show()


error = []
for i in range(len(b)):
    y_pred = (A[i]**2) * xstar[2] + A[i] * xstar[1] + xstar[0]
    error.append(np.square(y_pred - b[i]))

print("Mean squared error:", np.sum(error)/len(b))
