__author__ = "vaibhawvipul"

import pandas as pd
from numpy import random

dataframe = pd.read_csv("Height_Weight_single_variable_data_101_series_1.0.csv")

train_dataframe = dataframe[0:20]
test_dataframe = dataframe[20:35]

x_values = train_dataframe.loc[:,"Weight"].as_matrix()
y_values = train_dataframe.loc[:,"Height"].as_matrix()

test_input = test_dataframe.loc[:,"Weight"].as_matrix()
test_output = test_dataframe.loc[:,"Height"].as_matrix()

class NeuralNetwork:
    def __init__(self):
        random.seed(1)
        self.learning_rate = 0.001
        self.m,self.b = random.rand(2) #generating random slope and bias

    def linear_equation(self,m,x,b):
        return (m*x)+b

    def compute_error(self,training_output,output):
        Error = 0
        for i in range(0,len(training_output)):
            Error += ((training_output[i] - output[i]) ** 2)/len(training_output)
        return Error

    def train(self,training_set_inputs, training_set_outputs,epochs):
        for iterations in range(epochs):
            for i in range(0,len(training_set_inputs)):
                output = self.predict(training_set_inputs[i])
                b_gradient = (-2 * (training_set_outputs[i] - output))/len(training_set_inputs)
                m_gradient = (-2 * training_set_inputs[i] * (training_set_outputs[i] - output))/len(training_set_inputs)
                self.b = self.b - (self.learning_rate * b_gradient)
                self.m = self.m - (self.learning_rate * m_gradient)
                #print self.m, self.b, output, training_set_outputs[i]
        return [self.m,self.b]

    def predict(self,input):
        return self.linear_equation(self.m,input,self.b)

neural_network = NeuralNetwork()
print "starting with m = "+str(neural_network.m)+" and b = "+str(neural_network.b)+"\n"
output =[]
for i in range(0,len(x_values)):
    output.append(neural_network.predict(x_values[i]))
print "Loss function is : "+ str(neural_network.compute_error(y_values,output))+"\n"


neural_network.train(x_values,y_values,30000)

print "New m = "+str(neural_network.m)+" and b = "+str(neural_network.b)+"\n"

output =[]
for i in range(0,len(test_input)):
    output.append(neural_network.predict(test_input[i]))

for i in range(0,len(test_output)):
    print str(output[i])+" "+ str(test_output[i])

print "Loss function is : "+ str(neural_network.compute_error(test_output,output))