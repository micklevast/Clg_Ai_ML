import math
Dataset ={
"W15":0.3,
"W16":0.1,
"W25":-0.2,
"W26":0.4,
"W35":0.2,
"W36":-0.3,
"W45":0.1,
"W46":0.4,
"W57":-0.3,
"W67":0.2,
"05":0.2,
"06":0.1,
"07":-0.3
}
alpha = 0.8
def perceptron(Dataset,num_or_iteration,Y,INPUT,alpha,j):
    print("Running Iteration ->", j)
    X5 =Dataset["W15"]*INPUT[0]+Dataset["W25"]*INPUT[1]+Dataset["W35"]*INPUT[2]+Dataset["W45"]*INPUT[3]+Dataset["05"]
    X6 =Dataset["W16"]*INPUT[0]+Dataset["W26"]*INPUT[1]+Dataset["W36"]*INPUT[2]+Dataset["W46"]*INPUT[3]+Dataset["06"]
    
#    formulae------->
    O5 = 1/(1+math.exp(-X5))
    O6 = 1/(1+math.exp(-X6))
    # print("O5",O5,"O6",O6)
    X7 = Dataset["W57"]*O5+Dataset["W67"]*O6+Dataset["07"]
    O7 = 1/(1+math.exp(-X7))
    # print("O7",O7)
    
    Error7 = O7*(1-O7)*(Y-O7)
    Error6 = O6*(1-O6)*Error7*Dataset["W67"]
# -------------------------above are formulae
    
    Dataset["W67"] += alpha*Error7*O6
    Error5 = O5*(1-O5)*Error7*Dataset["W57"]
    
    Dataset["W57"] += alpha*Error7*O5
    #    formulae------->
    Dataset["05"] = Dataset["05"]+alpha * Error5
    Dataset["06"] = Dataset["06"]+alpha * Error6
    Dataset["07"] = Dataset["07"]+alpha * Error7
    Dataset["W15"] +=alpha*Error5*INPUT[0] 
    Dataset["W25"] +=alpha*Error5*INPUT[1] 
    Dataset["W35"] +=alpha*Error5*INPUT[2] 
    Dataset["W45"] +=alpha*Error5*INPUT[3] 
    j+=1
    Dataset["W16"] +=alpha*Error6*INPUT[0] 
    Dataset["W26"] +=alpha*Error6*INPUT[1] 
    Dataset["W36"] +=alpha*Error6*INPUT[2] 
    Dataset["W46"] +=alpha*Error6*INPUT[3] 
    print("The ERROR values in the iteration: ")
    print("ERROR7 = ",Error7,"ERROR6",Error6,"ERROR5",Error5)
    print("Final Error Result : ",1-O7)
Y=1
num_or_iteration=0
INPUT = [1,1,0,1]
num_or_iteration = int(input("Enter the number of num_or_iteration:"))
print(num_or_iteration)

for i in range(num_or_iteration):
    perceptron(Dataset,num_or_iteration,Y,INPUT,alpha,i+1)