
import pandas as pd
import numpy as np

from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dense, Dropout, Flatten, Activation
from keras.models import Sequential
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, Callback
from keras.optimizers import Adam
from keras import backend as k

import os

from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix,roc_curve,auc,log_loss
from sklearn.model_selection import train_test_split
import itertools
import shutil
import matplotlib.pyplot as plt
from PIL import Image

IMAGE_SIZE = 96
IMAGE_CHANNELS = 3

df_data = pd.read_csv('./input/train_labels.csv')
print(df_data.shape)
#image is black
df_data[df_data['id'] != '9369c7278ec8bcc6c880d99194de09fc2bd4efbe']

#causes a training error
df_data[df_data['id'] != 'dd6dfed324f9fcb6f93f46f32fc800f2ec196be2']