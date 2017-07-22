import csv
import cv2
import numpy as np
import matplotlib.pyplot as plt


def compare_images(left_image, right_image):
    print(type(left_image))
    print(left_image.shape)
    print(type(right_image))
    print(right_image.shape)
    
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
    f.tight_layout()
    ax1.imshow(cv2.cvtColor(left_image, cv2.COLOR_BGR2RGB))
    ax1.set_title('Shape '+ str(left_image.shape),
                  fontsize=50)
    ax2.imshow(cv2.cvtColor(right_image, cv2.COLOR_RGB2BGR))
    #cv2.imshow('image', right_image)
    ax2.set_title('Shape '+ str(right_image.shape)
                  , fontsize=50)
    plt.show()

def compare_cv_images(left_image, right_image):
    #cv2.imshow('image',left_image)
    cv2.imshow('image', right_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def load_training_data():
    lines = []
    images = []
    measurements = []

    with open('./data/driving_log.csv') as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            lines.append(line)

    for line in lines:
        source_path = line[0]
        filename = source_path.split('/')[-1]
        current_path = './data/IMG/' + filename
        image = cv2.imread(current_path)
        images.append(image)
        measurement = float(line[3])
        measurements.append(measurement)

    X_train = np.array(images)
    y_train = np.array(measurements)
    return X_train, y_train

def plot_loss_histogram(history_object):  
    plt.plot(history_object.history['loss'])
    plt.plot(history_object.history['val_loss'])
    plt.title('model mean squared error loss')
    plt.ylabel('mean squared error loss')
    plt.xlabel('epoch')
    plt.legend(['training set', 'validation set'], loc='upper right')
    plt.savefig('images/historgram.png')
#    plt.show()

from keras.models import Sequential, Model
from keras import backend as K
from keras.layers import Flatten, Dense, Lambda
from keras.layers.convolutional import Convolution2D, Cropping2D
from keras.layers.pooling import MaxPooling2D

def network_model(X_train, y_train):
    
    model = Sequential()
    #add normalization
    model.add(Lambda(lambda x: x /255.0 - 0.5 , input_shape=(160, 320, 3)))

    #model.add(Cropping2D(cropping=((70,25),(0,0))))
    #cropping_output = K.function([model.layers[1].input],
    #                                  [model.layers[1].output])
    
    #cropped_image = cropping_output(image[None,...]])[0]
    
    #compare_images(image,cropped_image.reshape(cropped_image.shape[1:]))
    
    #compare_cv_images(image,cropped_image)
    
    
    model.add(Convolution2D(24,5,5, subsample=(2,2), activation="relu"))
    model.add(Convolution2D(36,5,5, subsample=(2,2), activation="relu"))
    model.add(Convolution2D(48,5,5, subsample=(2,2), activation="relu"))
    model.add(Convolution2D(64,3,3, activation="relu"))
    model.add(Convolution2D(64,3,3, activation="relu"))
    model.add(Flatten())
    model.add(Dense(100))
    model.add(Dense(50))
    model.add(Dense(10))
    model.add(Dense(1))
    
    model.compile(loss='mse', optimizer='adam')
    history_object = model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=20)
    plot_loss_histogram(history_object)
    model.save('model.h5')
    
#Main()  
X_train, y_train = load_training_data()

network_model(X_train, y_train)
