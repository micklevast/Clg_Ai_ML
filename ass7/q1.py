# importing Python library

# --------->imp link to leaarn perceptron   |^|
# https://medium.com/@stanleydukor/neural-representation-of-and-or-not-xor-and-xnor-logic-gates-perceptron-algorithm-b0275375fea1

import numpy as np

# define Unit Step Function
def Avtivation_func(v):
	if v >= 0:
		return 1
	else:
		return 0

# design Perceptron Model
def perceptronModel(x, w, b):
	v = np.dot(w, x) + b
	y = Avtivation_func(v)
	return y

# NOT Logic Function
# wNOT = -1, bNOT = 0.5
def NOT_logicFunction(x):
    wNOT = -1
    # bNOT = 0.4
    bNOT = -1
    return perceptronModel(x, wNOT, bNOT)

# OR Logic Function
# w1 = 1, w2 = 1, bOR = -0.5
def OR_logicFunction(x):
    w = np.array([0.3, -0.2])
    bOR = 0.4
    # bOR = -1
    return perceptronModel(x, w, bOR)

# NOR Logic Function
# with OR and NOT
# function calls in sequence
def NOR_logicFunction(x):
    
    output_OR = OR_logicFunction(x)
    print("or:",output_OR)
    
    output_NOT = NOT_logicFunction(output_OR)
    print("not:",output_NOT)
    return output_NOT

# testing the Perceptron Model
test2 = np.array([0, 1])
test3 = np.array([1, 0])
test4 = np.array([1, 1])
test1 = np.array([0, 0])

print("NOR({}, {}) = {}".format(0, 1, NOR_logicFunction(test2)))
print("NOR({}, {}) = {}".format(1, 0, NOR_logicFunction(test3)))
print("NOR({}, {}) = {}".format(1, 1, NOR_logicFunction(test4)))
print("NOR({}, {}) = {}".format(0, 0, NOR_logicFunction(test1)))
