from __future__ import absolute_import, division, print_function
import pandas as pd
from numpy import *
import matplotlib.pyplot as plt

#data source https://github.com/Dataweekends/zero_to_deep_learning_udemy/blob/master/data/weight-height.csv
#reading dataframe
dataframe = pd.read_csv("weight-height.csv")

#Extracting data which has height weight information only for men.
dataframe = dataframe[0:5000]
x = dataframe["Height"]
y = dataframe["Weight"]

#converting dataframe to numpy array
height_array = x.values
weight_array = y.values

#train_set 75% of 5000
x_train = height_array[0:3750]
y_train = weight_array[0:3750]
print(x_train)
print(y_train)
#test_set 25% of dataset
x_test = height_array[3750:5000]
y_test = weight_array[3750:5000]
print(x_test)
print(y_test)

#visualize the data
plt.scatter(x_train,y_train)
plt.show()
#By looking at the visualization we can predict that
#Linear Regression will be a good fit to the data

def loss_function(b,m,x_val,y_val):
    totalError = 0
    for i in range(0,len(x_val)):
        x = x_val[i]
        y = y_val[i]
        totalError += (y - (m * x + b)) ** 2
    return totalError / float(len(x_val))

def step_gradient(b_current,m_current,x_val,y_val,learning_rate):
    b_gradient = 0
    m_gradient = 0
    N = float(len(x_val))
    for i in range(0, len(x_val)):
        x = x_val[i]
        y = y_val[i]
        b_gradient += -(2 / N) * (y - ((m_current * x) + b_current))
        m_gradient += -(2 / N) * x * (y - ((m_current * x) + b_current))
    new_b = b_current - (learning_rate * b_gradient)
    new_m = m_current - (learning_rate * m_gradient)
    return [new_b, new_m]

def batch_gradient_descent_executor(x_val, y_val, starting_b,starting_m,learning_rate,epochs):
    b = starting_b
    m = starting_m
    for i in range(epochs):
        b,m =step_gradient(b,m,x_val,y_val,learning_rate)
    return [b,m]

def main():
    random.seed(1) #so that program initializes same random numbers everytime
    initial_b, initial_m = random.random(2) #starting with random b and m
    epochs = 1000
    learning_rate = 0.0001
    print("Starting gradient descent at b = {0}, m = {1}, error = {2}".format(initial_b, initial_m,
                                                                              loss_function(
                                                                              initial_b, initial_m, x_test, y_test
                                                                              )))
    print("...Starting Training...")
    [b,m]= batch_gradient_descent_executor(x_train,y_train, initial_b, initial_m, learning_rate, epochs)
    print("After {0} epochs b = {1}, m = {2}, error = {3}".format(epochs, b, m,
                                                                      loss_function(b, m, x_test, y_test)))
    y_pred = []
    for i in x_test:
        y_pred.append(m*i+b)

    #visualize the predictions
    plt.scatter(x_test,y_test)
    plt.plot(x_test,y_pred)
    plt.show()

if __name__ == '__main__':
    main()
