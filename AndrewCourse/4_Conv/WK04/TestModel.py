from keras.models import Sequential
from keras.layers import Conv2D, ZeroPadding2D, Activation, Input, concatenate
from keras.models import Model
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import MaxPooling2D, AveragePooling2D
from keras.layers.merge import Concatenate
from keras.layers.core import Lambda, Flatten, Dense
from keras.initializers import glorot_uniform
from keras.engine.topology import Layer
from keras import backend as K
# 通道前置设置
K.set_image_data_format('channels_first')
import cv2
import os
import numpy as np
from numpy import genfromtxt
import pandas as pd
import tensorflow as tf
# 提供的辅助库
from fr_utils import *
from inception_blocks_v2 import *

#%matplotlib inline
#%load_ext autoreload
#%autoreload 2

np.set_printoptions(threshold=np.nan)


def triplet_loss(y_true, y_pred, alpha=0.2):
    """
    输入：
    -  y_true -- 标签，这个是keras规定要写的，不一定要用到
    -  y_pred :是一个列表，计算的128维的结果
    """

    anchor, positive, negative = y_pred[0], y_pred[1], y_pred[2]

    # 计算A和P的距离差值
    pos_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, positive)))
    # 计算A和N的距离差值
    neg_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, negative)))
    # 计算以上两个差值就上a的值
    basic_loss = tf.add(tf.subtract(pos_dist, neg_dist), alpha)
    # 计算大于0的值的和
    loss = tf.reduce_sum(tf.maximum(basic_loss, 0.))

    return loss


# 模型的创建
FRmodel = faceRecoModel(input_shape=(3, 96, 96))

FRmodel.compile(optimizer = 'adam', loss = triplet_loss, metrics = ['accuracy'])
load_weights_from_FaceNet(FRmodel)




