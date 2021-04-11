from random import random
from google.protobuf.descriptor import Descriptor
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras import layers, Sequential, regularizers
from tensorflow.python.keras import activations
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split

print(tf.__version__)

OUTPUT_DIR = 'output_dir'
N_EPOCHS = 500


def load_dataset():
    N_SAMPLES = 1000
    TEST_SIZE = None

    X,y = make_moons(n_samples =N_SAMPLES,noise = 0.25,random_state=100)
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=TEST_SIZE,random_state=42)
    return X,y,X_train,X_test,y_train,y_test


def make_plot(X, y, plot_name, file_name, XX=None, YY=None, preds=None, dark=False, output_dir=OUTPUT_DIR):
    # 绘制数据集的分布， X 为 2D 坐标， y 为数据点的标签
    if dark:
        plt.style.use('dark_background')

    axes = plt.gca()
    axes.set_xlim([-2, 3])
    axes.set_ylim([-1.5, 2])
    axes.set(xlabel="$x_1$", ylabel="$x_2$")
    plt.title(plot_name, fontsize=20, fontproperties='SimHei')
    plt.subplots_adjust(left=0.20)
    plt.subplots_adjust(right=0.80)
    if XX is not None and YY is not None and preds is not None:
        plt.contourf(XX, YY, preds.reshape(XX.shape), 25, alpha=0.08, cmap=plt.cm.Spectral)
        plt.contour(XX, YY, preds.reshape(XX.shape), levels=[.5], cmap="Greys", vmin=0, vmax=.6)
    # 绘制散点图，根据标签区分颜色m=markers
    markers = ['o' if i == 1 else 's' for i in y.ravel()]
    # mscatter(X[:, 0], X[:, 1], c=y.ravel(), s=20, cmap=plt.cm.Spectral, edgecolors='none', m=markers, ax=axes)
    # 保存矢量图
    plt.savefig(output_dir + '/' + file_name)
    plt.close()


def network_layer_influence(X_train,y_train):

    for n in range(5):
        model = Sequential()

        model.add(layers.Dense(8,input_dim = 2,activation='relu'))

        for _ in range(n):
            model.add(layers.Dense(32,activation='relu'))
        
        model.add(layers.Dense(1,activations='sigmoid'))

        model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
        model.fit(X_train,y_train,epochs=N_EPOCHS,verbose=1)

        xx = np.arange(-2,3,0.01)
        yy = np.arange(-1.5,2,0.01)

        XX,YY = np.meshgrid(xx,yy)

        preds = model.predict_classes(np.c_[XX.ravel(),YY.ravel()])
        title = "网络层数：{0}".format(2 + n)
        file = "网络容量_%i.png" % (2 + n)
        make_plot(X_train, y_train, title, file, XX, YY, preds, output_dir=OUTPUT_DIR + '/network_layers')


def main():
    X,y,X_train,X_test,y_train,y_test = load_dataset()

    # 绘制数据集分布
    make_plot(X, y, None, "月牙形状二分类数据集分布.svg")


     # 网络层数的影响
    network_layer_influence(X_train, y_train)


if __name__ == '__main__':
    main()





        