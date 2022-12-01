# Fitting Logistic Regression to the training set
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

input = np.array([0.5, 1.0, 1.25, 2.5, 3.0, 1.75, 4.0, 4.25, 4.75, 5.0])
input = input.reshape(-1, 1)
output = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])


# z = np.array([[1, 2, 3, 4],
#          [5, 6, 7, 8],
#          [9, 10, 11, 12]])
# z.shape
# # Now trying to reshape with (-1) . Result new shape is (12,) and is compatible with original shape (3,4)
# z.reshape(-1)
# # array([ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12])
# # Now trying to reshape with (-1, 1) . We have provided column as 1 but rows as unknown . So we get result new shape as (12, 1).again compatible with original shape(3,4)
#  to use reshape(1,-1) for a single sample; i.e. single row
# New shape (2, -1). Row 2, column unknown. we get result new shape as (2,6)
# z.reshape(1,-1)


model = LogisticRegression(solver='liblinear', random_state=0)
model.fit(input, output)

LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
                   intercept_scaling=1, l1_ratio=None, max_iter=100,
                   multi_class='warn', n_jobs=None, penalty='l2',
                   random_state=0, solver='liblinear', tol=0.0001, verbose=0,
                   warm_start=False)

model = LogisticRegression(solver='liblinear',
                           random_state=0).fit(input, output)


# You can quickly get the attributes of your model. For example, the attribute .classes_ represents the array of distinct values that y takes:
model.classes_
print(model.classes_)


# This is the example of binary classification, and y can be 0 or 1, as indicated above.
# You can also get the value of the slope 𝑏₁ and the intercept 𝑏₀ of the linear function 𝑓 like so:

model.intercept_

model.coef_

prob_of_beloginess = model.predict_proba(input)
ans = model.predict(input)
print(prob_of_beloginess)
print(ans)


# When you have nine out of ten observations classified correctly, the accuracy of your model is equal to 9/10=0.9, which you can obtain with .score():
model.score(input, output)
print("score:", model.score(input, output))
