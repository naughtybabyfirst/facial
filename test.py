from train import emotion_analysis, reshape_dataset
import matplotlib.pyplot as plt
import numpy as np
from keras.preprocessing import image
from model import build_model

if __name__ == '__main__':
    # path = '/home/jing/PycharmProjects/facial/dataset/fer2013/fer2013.csv'
    num_classes = 7

    # x_train, y_train, x_test, y_test = reshape_dataset(path, num_classes)

    model = build_model(num_classes)
    model.load_weights('/home/jing/PycharmProjects/facial/model_checkpoints/facial_expression_model_weights.h5')

    # monitor_testset_results = False
    #
    # if monitor_testset_results == True:
    #     # make predictions for test set
    #     predictions = model.predict(x_test)
    #
    #     index = 0
    #     for i in predictions:
    #         if index < 30 and index >= 20:
    #             # print(i) #predicted scores
    #             # print(y_test[index]) #actual scores
    #
    #             testing_img = np.array(x_test[index], 'float32')
    #             testing_img = testing_img.reshape([48, 48])
    #
    #             plt.gray()
    #             plt.imshow(testing_img)
    #             plt.show()
    #
    #             print(i)
    #
    #             emotion_analysis(i)
    #             print("----------------------------------------------")
    #         index = index + 1

    # ------------------------------
    # make prediction for custom image out of test set

    # img = image.load_img("/home/jing/PycharmProjects/facial/dataset/pablo.png", grayscale=True, target_size=(48, 48))
    # img = image.load_img("/home/jing/PycharmProjects/facial/dataset/monalisa.png", grayscale=True, target_size=(48, 48))
    img = image.load_img("/home/jing/PycharmProjects/facial/dataset/jackman.png", grayscale=True, target_size=(48, 48))

    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)

    x /= 255

    custom = model.predict(x)
    t1 = emotion_analysis(custom[0])

    x = np.array(x, 'float32')
    x = x.reshape([48, 48])
    plt.gray()

    plt.imshow(x)
    plt.show()
    # ------------------------------
