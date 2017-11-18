__author__ = "vaibhawvipul"

#importing libraries
import numpy as np
import matplotlib.pypolot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.examples.tutorials.mnist import input_data

# Importing fashion mnist

fashion_mnist = input_data.read_data_sets("input/data",one_hot=True)
