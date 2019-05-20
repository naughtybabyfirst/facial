import numpy as np
import keras
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from model import build_model
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd


def read_data_np(path):
    with open(path) as f:
        content = f.readlines()

    lines = np.array(content)

    num_of_instances = lines.size
    print("number of instances: ", num_of_instances)
    print("instance length: ", len(lines[1].split(",")[1].split(" ")))

    return lines, num_of_instances


def read_data_pd(path):

    data_df = pd.read_csv(path, header=0)
    lines = len(data_df)
    print(data_df.head())

    return data_df, lines


def reshape_dataset(paths, num_classes):
    x_train, y_train, x_test, y_test = [], [], [], []

    lines, num_of_instances = read_data_np(paths)
    # data_df, lines = read_data_pd(paths)

    # ------------------------------
    # transfer train and test set data
    for i in range(1, num_of_instances):
        try:
            emotion, img, usage = lines[i].split(",")
            # emotion, img, usage = data_df['emotion'][i], data_df['pixels'][i], data_df['Usage'][i]

            val = img.split(" ")

            pixels = np.array(val, 'float32')

            emotion = keras.utils.to_categorical(emotion, num_classes)

            if 'Training' in usage:
                y_train.append(emotion)
                x_train.append(pixels)
            elif 'PublicTest' in usage:
                y_test.append(emotion)
                x_test.append(pixels)
        except:
            print("", end="")

    # ------------------------------
    # data transformation for train and test sets
    x_train = np.array(x_train, 'float32')
    y_train = np.array(y_train, 'float32')
    x_test = np.array(x_test, 'float32')
    y_test = np.array(y_test, 'float32')

    x_train /= 255  # normalize inputs between [0, 1]
    x_test /= 255

    x_train = x_train.reshape(x_train.shape[0], 48, 48, 1)
    x_train = x_train.astype('float32')
    x_test = x_test.reshape(x_test.shape[0], 48, 48, 1)
    x_test = x_test.astype('float32')

    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')

    y_train = y_train.reshape(y_train.shape[0], 7)
    y_train = y_train.astype('int16')
    y_test = y_test.reshape(y_test.shape[0], 7)
    y_test = y_test.astype('int16')

    print('--------x_train.shape:', x_train.shape)
    print('--------y_train.shape:', y_train.shape)


    print(len(x_train), 'train x size')
    print(len(y_train), 'train y size')
    print(len(x_test), 'test x size')
    print(len(y_test), 'test y size')

    return x_train, y_train, x_test, y_test


# def batch_process(path, batch_size):
#     x_train, y_train, x_test, y_test = reshape_dataset(path)
#     gen = ImageDataGenerator()
#     train_generator = gen.flow(x_train, y_train, batch_size=batch_size)
#
#     return x_train, y_train, x_test, y_test, train_generator


# def compile_model(models):
#     model = models.compile(loss='categorical_crossentropy'
#                   , optimizer=keras.optimizers.Adam()
#                   , metrics=['accuracy']
#                   )
#     return model


def emotion_analysis(emotions):
    objects = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')
    y_pos = np.arange(len(objects))

    plt.bar(y_pos, emotions, align='center', alpha=0.5)
    plt.xticks(y_pos, objects)
    plt.ylabel('percentage')
    plt.title('emotion')
    plt.show()
    return plt


if __name__ == '__main__':
    num_classes = 7  # angry, disgust, fear, happy, sad, surprise, neutral
    batch_size = 256
    epochs = 5

    config = tf.ConfigProto(device_count={'GPU': 0, 'CPU': 56})  # max: 1 gpu, 56 cpu
    sess = tf.Session(config=config)
    keras.backend.set_session(sess)

    path = '/home/jing/PycharmProjects/facial/dataset/fer2013/fer2013.csv'
    x_train, y_train, x_test, y_test = reshape_dataset(path, num_classes)

    gen = ImageDataGenerator()
    train_generator = gen.flow(x_train, y_train, batch_size=batch_size)


    # for data_batch, label_batch in train_generator:
    #     print('data_batch:', data_batch.shape)
    #     print('label_batch:', label_batch.shape)

    m = build_model(num_classes)
    print('model:', m)
    print('train_generator:', train_generator)
    # m = compile_model(m)
    m.compile(loss='categorical_crossentropy'
                   , optimizer=keras.optimizers.Adam()
                   , metrics=['accuracy']
                   )
    print('m:', m)
    m.fit_generator(train_generator, steps_per_epoch=batch_size, epochs=epochs)

    m.save('/home/jing/PycharmProjects/facial/model_checkpoints/facial_expression_model_weights.h5')
    print('save weight..')
