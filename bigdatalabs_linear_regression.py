from __future__ import absolute_import, division, print_function
import pandas as pd
from numpy import *
from sklearn import linear_model
import matplotlib.pyplot as plt

#reading dataframe
dataframe = pd.read_csv("weight-height.csv")

#Extracting data which has height weight information only for men.
dataframe = dataframe[0:5000]
x = dataframe[["Height"]]
y = dataframe[["Weight"]]

#converting dataframe to numpy array
height_array = x.values
weight_array = y.values

#train_set 75% of 5000
x_train = height_array[0:3750]
y_train = weight_array[0:3750]

#test_set 25% of dataset
x_test = height_array[3750:5000]
y_test = weight_array[3750:5000]


#visualize the data
plt.scatter(x_train,y_train)
plt.show()
#By looking at the visualization we can predict that
#Linear Regression will be a good fit to the data

def main():

    reg = linear_model.LinearRegression()
    reg.fit(x_train,y_train)

    #visualize the predictions
    plt.scatter(x_test,y_test)
    #plt.plot(x_test,y_pred)
    plt.plot(x_test,reg.predict(x_test))
    plt.show()

if __name__ == '__main__':
    main()
