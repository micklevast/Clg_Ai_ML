import numpy as np

# define Unit Step Function
def Avtivation_func(v):
	if v >= 0:
		return 1
	else:
		return 0

w = [0.3,-0.2]
b = 0.4
alpha = 0.2
def perceptron(x,w,b,val,alpha):
    sum = np.dot(w,x)+b
    ans = Avtivation_func(sum)
    err = val-ans
    for j in range(len(x)):
        w[j]=w[j]+alpha * err *x[j]
        
    return err

err = 1
num_of_iteration=0

Xor_set = [
[0,0,1],
[0,1,0],
[1,0,0],
[1,1,0]
]


# Training->
while(err):
    err = 0
    for i in range(len(Xor_set)):
        x = Xor_set[i][:-1]  #array of first two value from first list
        real_outPut = Xor_set[i][-1]    # only last value as realOutput
        ans = perceptron(x,w,b,real_outPut,alpha)
        err = err or ans
        num_of_iteration+=1


# Testing->
print("-----------after training Updated Weight will be -------------")
print(w)
print("final number of iteration for to update weight value : ",num_of_iteration-1)
print("--------------------------------------------------------------")


# -----------------------------testing with real output
def perceptronModel(x, w, b):
	v = np.dot(w, x) + b
	y = Avtivation_func(v)
	return y

# NOT Logic Function
# wNOT = -1, bNOT = 0.5
def NOT_logicFunction(x):
    wNOT = 1
    # bNOT = 0.4
    bNOT = -0.5
    return perceptronModel(x, wNOT, bNOT)

# OR Logic Function
# w1 = 1, w2 = 1, bOR = -0.5
def OR_logicFunction(x):
    # w = np.array([0.3, -0.2])
    # w=w;
    w=[-0.5, -0.6000000000000001]
    bOR = 0.4
    # bOR = -1
    return perceptronModel(x, [-0.5, -0.6000000000000001], bOR)

# NOR Logic Function
# with OR and NOT
# function calls in sequence
def NOR_logicFunction(x):
    
    output_OR = OR_logicFunction(x)
    # print("or:",output_OR)
    
    output_NOT = NOT_logicFunction(output_OR)
    # print("not:",output_NOT)
    return output_NOT

# testing the Perceptron Model

for i in range(len(Xor_set)):
    input_x1_x2 = Xor_set[i][:-1]  #array of first two value from first list
    print("NOR({}, {}) = {}".format(Xor_set[i][0],Xor_set[i][1], NOR_logicFunction([Xor_set[i][0],Xor_set[i][1]])))