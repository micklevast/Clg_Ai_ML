from sklearn.metrics import mean_squared_error
import numpy as np


class Simple_Linear_Regression():

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

    def calculate(self, x):
        """
        Main Method:
            Args:
            x (float) : value of feature
            Return:
            output value of class y(float)
        """
        return round((self.a1 * x + self.a0), 2)

# inputs = [6.2, 6.5, 5.4, 6.5, 7.1, 7.9, 8.5, 8.9, 9.5, 10.6]
# outputs = [26.3, 26.6, 25, 26, 27.9, 30.4, 35.4, 38.5, 42.6, 48.3]


inputs = [1, 2, 3, 4, 5]
outputs = [1.2, 1.8, 2.6, 3.2, 3.8]

x = 3
slr = Simple_Linear_Regression(inputs, outputs)
print("Predicted value of :", x, " is =", slr.calculate(x))

predicted_output = [slr.calculate(inputs[i]) for i in range(len(inputs))]
# mean_squared_error(Y_true,Y_pred)
print("error:", mean_squared_error(outputs, predicted_output))
print("accuracy:", 100-(mean_squared_error(outputs, predicted_output)*100))
