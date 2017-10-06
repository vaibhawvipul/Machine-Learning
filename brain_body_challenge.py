#First Deep Learning Code!
#Reference Siraj Raval Intro to Deep Learning video 1

import pandas as pd
from sklearn import linear_model
import matplotlib.pyplot as plt

#read_data

dataframe = pd.read_csv("../brain_body_challenge.txt")
print dataframe #print dataframe

x_values = dataframe[["x"]]
y_values = dataframe[["y"]]

print x_values

#train model on data

reg = linear_model.LinearRegression()

reg.fit(x_values,y_values)

#visualize the data
plt.scatter(x_values,y_values)
plt.plot(x_values,reg.predict(x_values))
plt.show()

print "Ridge"

#Ridge linear_model
reg = linear_model.Ridge(alpha = 0.5)
reg.fit (x_values.as_matrix(),y_values.as_matrix())
plt.scatter(x_values,y_values)
plt.plot(x_values,reg.predict(x_values.as_matrix()))
plt.show()