import numpy as np
import pandas as pd
import os
from pprint import pprint
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K

train_data = pd.read_csv('emnist-balanced-train.csv', header=None)
test_data = pd.read_csv('emnist-balanced-test.csv', header=None)
print(train_data.shape, test_data.shape)