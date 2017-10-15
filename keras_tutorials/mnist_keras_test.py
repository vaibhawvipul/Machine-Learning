import matplotlib.pyplot as plt
import numpy
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.utils import np_utils
from mnist_keras_tut_more_advance import model
from scipy.misc import imread, imsave, imresize

#model.load_weights('./mnistneuralnet.h5')

img_width, img_height = 28, 28

img = imread('./seven.png',mode = "L")
img = numpy.invert(img)
print img.shape[0]
img = imresize(img, (28,28))
img = img.reshape(28,28,1)
img = numpy.invert(img)/255
#prediction = model.predict(img)

print img
