import numpy as np
import math


class Logistic_Regression():

    """
        Class to calculate simple regression
    """

    def __init__(self, X, Y):
        """
            Constructor:
                Args :
                X (list) : input value of feature
                Y (list) : corresponding output values of feature.

        """
        self.input = np.array(X)
        self.output = np.array(Y)

        self.mean_x = sum(self.input)/len(self.input)
        self.mean_y = sum(self.output)/len(self.output)

        self.mean_sq_x = np.dot(self.input, self.input)/len(self.input)
        self.mean_xy = np.dot(self.input, self.output)/len(self.input)

        self.a1 = (self.mean_xy - self.mean_x*self.mean_y) / \
            ((self.mean_sq_x) - (self.mean_x)**2)

        self.a0 = self.mean_y - self.a1*self.mean_x

    def sigmoid(self, x):
        return 1/(1 + math.e**(-(self.a0 + self.a1*x)))

    def calculate(self, x):
        if (self.sigmoid(x) >= 0.5):
            return 1
        else:
            return 0

    def log_fun(self, input):
        # squares = [x*x for x in range(11)]

        # lst = [self.calculate(val) for val in self.input]
        lst = []

        for value in input:
            lst.append(self.calculate(value))
        return lst


input = np.array([0.5, 1.0, 1.25, 2.5, 3.0, 1.75, 4.0, 4.25, 4.75, 5.0])
output = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])


def my_accuracy(lst1, lst2):
    count_yes = 0
    for i in range(len(lst1)):
        if lst1[i] == lst2[i]:
            count_yes = count_yes+1
    return (count_yes)/len(lst1)

# model.score(input, output)
# print("score:", model.score(input, output))


x = 7
myObj = Logistic_Regression(input, output)
# print("Predicted value of ", x, " is ", slr.calculate(x))
pred_ans = myObj.log_fun(input)
print(myObj.log_fun(input))
print("accuracy:", my_accuracy(pred_ans, output))
