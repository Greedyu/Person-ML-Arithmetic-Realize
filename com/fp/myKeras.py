import gzip
import os
import struct
import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import SGD
from tensorflow.examples.tutorials.mnist import input_data


from keras.utils import np_utils





def load_mnist(path, kind):
    """Load MNIST data from `path`"""

    labels_path = os.path.join(path,
                               '%s-labels-idx1-ubyte'
                               % kind)
    images_path = os.path.join(path,
                               '%s-images-idx3-ubyte'
                               % kind)
    with open(labels_path, 'rb') as lbpath:
        magic, n = struct.unpack('>II',
                                 lbpath.read(8))
        labels = np.fromfile(lbpath,
                             dtype=np.uint8)

    with open(images_path, 'rb') as imgpath:
        magic, num, rows, cols = struct.unpack('>IIII',
                                               imgpath.read(16))
        images = np.fromfile(imgpath,
                             dtype=np.uint8).reshape(len(labels), 784)

    return images, labels


def load_mnist_test(path, kind):
    """Load MNIST data from `path`"""

    labels_path = os.path.join(path,
                               '%s-labels-idx1-ubyte'
                               % kind)
    images_path = os.path.join(path,
                               '%s-images-idx3-ubyte'
                               % kind)
    with open(labels_path, 'rb') as lbpath:
        magic, n = struct.unpack('>II',
                                 lbpath.read(8))
        labels = np.fromfile(lbpath,
                             dtype=np.uint8)

    with open(images_path, 'rb') as imgpath:
        magic, num, rows, cols = struct.unpack('>IIII',
                                               imgpath.read(16))
        images = np.fromfile(imgpath,
                             dtype=np.uint8).reshape(len(labels), 784)

    return images, labels

def read_data(path,kind):
    with gzip.open(path + kind + "-labels-idx1-ubyte.gz") as flbl:
        magic, num = struct.unpack(">II",flbl.read(8))
        label = np.fromstring(flbl.read(),dtype=np.int8)
    with gzip.open(path + kind + "-images-idx3-ubyte.gz",'rb') as fimg:
        magic, num, rows, cols = struct.unpack(">IIII",fimg.read(16))
        image = np.fromstring(fimg.read(),dtype=np.uint8).reshape(len(label),rows,cols)
    return (label, image)


def to4d(img):
    return img.reshape(img.shape[0], 784).astype(np.float32) / 255


if __name__ == '__main__':

    model = Sequential()  # 顺序模型
    # model.add(Dense(units=64,input_dim=100)) #输入层
    # model.add(Activation("relu"))   #激活函数是relu
    # model.add(Dense(units=10))
    # model.add(Activation("softmax"))
    # model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
    # model.compile(loss='categorical_crossentropy', optimizer=SGD(lr=0.01, momentum=0.9, nesterov=True))

    model.add(Dense(input_dim=28 * 28, output_dim=500))
    model.add(Activation('sigmoid'))
    model.add(Dense(output_dim=500))
    model.add(Activation('sigmoid'))
    model.add(Dense(output_dim=10))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy', optimizer=SGD(lr=0.1), metrics=['accuracy'])

    # x_train, y_train = load_mnist("/Users/dongsheng/Downloads/mnist", "train")
    # x_test, y_test = load_mnist("/Users/dongsheng/Downloads/mnist", "t10k")

    # (X_train, y_train), (X_test, y_test) = mnist.load_data()

    #选择你下载的包路径
    mnist = input_data.read_data_sets("/Users/dongsheng/Documents/me/resource/mnist/", one_hot=True)

    x_train, y_train = mnist.train.images, mnist.train.labels
    x_test, y_test = mnist.test.images, mnist.test.labels
    # x_train = x_train.reshape(-1, 784).astype('float32')
    # x_test = x_test.reshape(-1, 784).astype('float32')

    model.fit(x_train, y_train, batch_size=100, nb_epoch=10)

    loss_and_metrics = model.evaluate(x_test, y_test)
    model.predict()
    print('Total loss on Testing Set:', loss_and_metrics[0])
    print('Accuracy of  Testing Set:', loss_and_metrics[1])
