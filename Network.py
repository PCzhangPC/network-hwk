import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
import keras.callbacks


class DataReader:
    def __init__(self, im_path, la_path):
        self.im_path = im_path
        self.la_path = la_path

    def __read_one_image(self, name):
        image_path = self.im_path + name
        im = np.array(Image.open(image_path))
        return im

    def __read_all_image(self, num):
        im_arr = []
        for i in range(num):
            name = r'\pic_' + str(i) + '.png'
            im_arr.append(self.__read_one_image(name))
        return np.array(im_arr)

    def __read_lables(self):
        with open(self.la_path, 'rb') as f:
            lable_arr = f.readlines()
            for i in range(len(lable_arr)):
                lable_arr[i] = int(lable_arr[i])
            return np.array(lable_arr)

    def read_im_and_lb(self, num):
        return self.__read_all_image(num), self.__read_lables()


class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = {'batch': [], 'epoch': []}
        self.accuracy = {'batch': [], 'epoch': []}
        self.val_loss = {'batch': [], 'epoch': []}
        self.val_acc = {'batch': [], 'epoch': []}

    def on_batch_end(self, batch, logs={}):
        self.losses['batch'].append(logs.get('loss'))
        self.accuracy['batch'].append(logs.get('acc'))
        self.val_loss['batch'].append(logs.get('val_loss'))
        self.val_acc['batch'].append(logs.get('val_acc'))

    def on_epoch_end(self, batch, logs={}):
        self.losses['epoch'].append(logs.get('loss'))
        self.accuracy['epoch'].append(logs.get('acc'))
        self.val_loss['epoch'].append(logs.get('val_loss'))
        self.val_acc['epoch'].append(logs.get('val_acc'))

    def loss_plot(self, loss_type):
        iters = range(len(self.losses[loss_type]))
        #创建一个图
        plt.figure()
        # acc
        plt.plot(iters, self.accuracy[loss_type], 'r', label='train acc')#plt.plot(x,y)，这个将数据画成曲线
        # loss
        plt.plot(iters, self.losses[loss_type], 'g', label='train loss')
        if loss_type == 'epoch':
            # val_acc
            plt.plot(iters, self.val_acc[loss_type], 'b', label='val acc')
            # val_loss
            plt.plot(iters, self.val_loss[loss_type], 'k', label='val loss')
        plt.grid(True)#设置网格形式
        plt.xlabel(loss_type)
        plt.ylabel('acc-loss')#给x，y轴加注释
        plt.legend(loc="upper right")#设置图例显示位置
        plt.show()

#创建一个实例LossHistory
history = LossHistory()


class CnnNetworkModel:
    def __init__(self, type_num):
        self.__construct_model(type_num)

    def __construct_model(self, type_num):
        self.model = Sequential()
        self.model.add(Convolution2D(32, 5, strides=1, border_mode='valid', input_shape=(1, 28, 28),
                                     data_format="channels_first", activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(0.2))
        self.model.add(Flatten())
        self.model.add(Dense(128, activation='relu'))
        self.model.add(Dense(type_num, activation='softmax'))
        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    def get_model(self):
        return self.model


class MLPNetworkModel:
    def __init__(self):
        self.__construct_model()

    def __construct_model(self):
        self.model = Sequential()
        self.model.add(Dense(28*28, input_dim=28*28, init='uniform', activation='relu'))
        self.model.add(Dense(10, init='uniform', activation='sigmoid'))
        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    def get_model(self):
        return self.model


def test_cnn(train_im, train_lb, test_im, test_lb):
    # cnn
    train_im = train_im.reshape(train_im.shape[0], 1, 28, 28).astype('float32')
    test_im = test_im.reshape(test_im.shape[0], 1, 28, 28).astype('float32')
    train_im = train_im / 255
    test_im = test_im / 255
    train_lb = np_utils.to_categorical(train_lb)
    test_lb = np_utils.to_categorical(test_lb)

    model = CnnNetworkModel(train_lb.shape[1]).get_model()
    model.fit(train_im, train_lb, validation_data=(test_im, test_lb), nb_epoch=10, batch_size=128, verbose=2, callbacks=[history])
    scores = model.evaluate(test_im, test_lb, verbose=0)
    print("CNN Error: %.2f%%" % (100 - scores[1] * 100))
    history.loss_plot('epoch')


def test_mlp(train_im, train_lb, test_im, test_lb):
    # mlp
    train_im = train_im.reshape(train_im.shape[0], 784)
    test_im = test_im.reshape(test_im.shape[0], 784)

    train_im = train_im / 255
    test_im = test_im / 255
    train_lb = np_utils.to_categorical(train_lb)
    test_lb = np_utils.to_categorical(test_lb)
    model = MLPNetworkModel().get_model()
    model.fit(train_im, train_lb, validation_data=(test_im, test_lb), epochs=10, verbose=2)
    scores = model.evaluate(test_im, test_lb, verbose=0)
    print("MLP Error: %.2f%%" % (100 - scores[1] * 100))



if __name__ == '__main__':
    print('in')
    train_dr = DataReader(im_path=r'C:\Users\Administrator\Desktop\train_set\pic',
                          la_path=r'C:\Users\Administrator\Desktop\train_set\lable\train_lables.txt')
    train_im, train_lb = train_dr.read_im_and_lb(60000)

    test_dr = DataReader(im_path=r'C:\Users\Administrator\Desktop\test_set\pic',
                         la_path=r'C:\Users\Administrator\Desktop\test_set\lable\test_lables.txt')
    test_im, test_lb = test_dr.read_im_and_lb(10000)

  #  test_mlp(train_im, train_lb, test_im, test_lb)
    test_cnn(train_im, train_lb, test_im, test_lb)
