import numpy as np

# Author -Sasidhar Reddy

def func_error(data,initial_b,initial_m):
    error=0
    for i in data:
        error+= ((initial_m * i[0] + initial_b) - i[1] ) **2
    return error/len(data)

def func_gradient(data,initial_b,initial_m,learning_rate,iterations):
    b=initial_b
    m=initial_m
    for x in range(iterations):
        for i in data:
            b -= (2/float(len(data))) * learning_rate * ((m * i[0] + b) - i[1])
            m -= (2/float(len(data))) * learning_rate * ((m * i[0] + b) - i[1]) * i[0]
    return (b,m)

def func_normal(data):
    temp=0
    x=[]
    y=[]
    for i in data:
        y.append(i[1])
        temp=i[0]
        i[0]=1
        i[1]=temp
        x.append(i)
    X=np.array(x)
    Y=np.array(y)
    
    return np.linalg.inv(X.T @ X) @ X.T @ Y        


def run():
    raw_data = np.genfromtxt('C:\\Users\\sasidh1x\\Desktop\\ML\\Q1\\featuresX.dat',delimiter=',')
    initial_b=0
    initial_m=0
    iterations=1000
    learning_rate=0.0001
    print(func_error(raw_data,initial_b,initial_m))
    print(func_gradient(raw_data,initial_b,initial_m,learning_rate,iterations))
    b,m = func_gradient(raw_data,initial_b,initial_m,learning_rate,iterations)
    print(func_error(raw_data,b,m))
    normal_output = func_normal(raw_data)
    print(normal_output)
    print(func_error(raw_data,normal_output[0],normal_output[1]))

run()