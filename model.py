import csv
import cv2
import numpy as np
import matplotlib.pyplot as plt

from random import shuffle

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

def load_image_data():
    lines = []
    images = []
    left_images = []
    right_images = []
    measurements = []

    with open('./data/driving_log.csv') as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            lines.append(line)

    for line in lines:
        source_path = line[0]
        left_source_path = line[1]
        right_source_path = line[2]
        
        filename = source_path.split('/')[-1]
        left_filename = left_source_path.split('/')[-1]
        right_filename = right_source_path.split('/')[-1]
        
        current_path = './data/IMG/' + filename
        left_current_path = './data/IMG/' + left_filename
        right_current_path = './data/IMG/' + right_filename
        image = cv2.imread(current_path)
        images.append(image)

        left_image = cv2.imread(left_current_path)
        left_images.append(image)
        right_image = cv2.imread(right_current_path)
        right_images.append(image)
        
        measurement = float(line[3])
        measurements.append(measurement)

    return images, left_images, right_images, measurements

def reduce_centered_images(reduce_percent, center_images, left_images, right_images, measurements):
    
    temp_center_images = []
    temp_left_images = []
    temp_right_images = []
    temp_measurements = []

    original_size = len(measurements)
    numb_zero_measurements = measurements.count(0.0)
    numb_to_remove = int(numb_zero_measurements * reduce_percent)
    
    print("Removing %d of %d centered image data\n" %(numb_to_remove, numb_zero_measurements))

    for index in range(original_size):
        if measurements[index] != 0.0:
            temp_center_images.append(center_images[index])
            temp_left_images.append(left_images[index])
            temp_right_images.append(right_images[index])
            temp_measurements.append(measurements[index])
        else:
            numb_to_remove = numb_to_remove - 1
            if numb_to_remove <= 0:
                temp_center_images.append(center_images[index])
                temp_left_images.append(left_images[index])
                temp_right_images.append(right_images[index])
                temp_measurements.append(measurements[index])

    return temp_center_images, temp_left_images, temp_right_images, temp_measurements
def add_side_cameras(steer_adjust, center_images, left_images, right_images, measurements):
    temp_center_images = []
    temp_measurements = []
    print("Adding Side Cameras\n")
    
    for index, measurement in enumerate(measurements):
        temp_center_images.append(center_images[index])
        temp_measurements.append(measurements[index])
        temp_center_images.append(left_images[index])
        temp_measurements.append(measurements[index]+steer_adjust)
        temp_center_images.append(right_images[index])
        temp_measurements.append(measurements[index]-steer_adjust)

    print("Pre Side Carmera data size: %d Post Side Carmera data size: %d\n" %(len(measurements), len(temp_measurements)))
    return temp_center_images, temp_measurements

def flip_images(center_images, measurements):
    temp_center_images = []
    temp_measurements = []

    print("Flipping Images\n")

    for image, measurement in zip(center_images, measurements):
        temp_center_images.append(image)
        temp_measurements.append(measurement)
        if measurement != 0.0:
            temp_center_images.append(cv2.flip(image,1))
            temp_measurements.append(measurement*-1.0)

    print("Pre Flip data size: %d Post Flip data size: %d\n" %(len(measurements), len(temp_measurements)))
    return temp_center_images, temp_measurements

def augment_images(center_images, left_images, right_images, measurements):
    center_images, left_images, right_images, measurements = reduce_centered_images(0.66, center_images, left_images, right_images, measurements)
    center_images, measurements = add_side_cameras(0.2, center_images, left_images, right_images, measurements)
    center_images, measurements = flip_images(center_images, measurements)
    return center_images, left_images, right_images, measurements

def plot_measurement_histogram(measurements):  
    plt.hist(measurements, bins=100)
    plt.title('Steering Angle Historgram')
    plt.ylabel('number of measurements')
    plt.xlabel('angle')
    plt.show()
    #plt.savefig('images/steering_hist.png')


from keras.models import Sequential, Model
from keras import backend as K
from keras.layers import Flatten, Dense, Lambda, Dropout
from keras.layers.convolutional import Convolution2D, Cropping2D
from keras.layers.pooling import MaxPooling2D

def network_model(X_train, y_train):
    
    model = Sequential()
    #add normalization
    model.add(Lambda(lambda x: x /255.0 - 0.5 , input_shape=(160, 320, 3)))

    model.add(Cropping2D(cropping=((70,25),(0,0))))
    #cropping_output = K.function([model.layers[1].input],
    #                                  [model.layers[1].output])
    
    #cropped_image = cropping_output(image[None,...]])[0]
    
    #compare_images(image,cropped_image.reshape(cropped_image.shape[1:]))
    
    #compare_cv_images(image,cropped_image)

    

    # Conv Layer #0 (depth=3, kernel=1x1) - change color space
    model.add(Convolution2D(3, 1, 1, border_mode='same'))
    model.add(Convolution2D(24,5,5, subsample=(2,2), activation="relu"))
#    model.add(MaxPooling2D(pool_size=(2,2)))
 #   model.add(Dropout(0.5))
    model.add(Convolution2D(36,5,5, subsample=(2,2), activation="relu"))
  #  model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Convolution2D(48,5,5, subsample=(2,2), activation="relu"))
   # model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Convolution2D(64,3,3, activation="relu"))
   # model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Convolution2D(64,3,3, activation="relu"))
   # model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Flatten())
    model.add(Dense(100))

    model.add(Dense(50))
    model.add(Dropout(0.5))
    model.add(Dense(10))
    model.add(Dense(1))
    
    model.compile(loss='mse', optimizer='adam')
    history_object = model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=10)
    model.save('model.h5')
    plot_loss_histogram(history_object)

def augment_data(X_train, y_train):
    new_X_train = X_train
    new_y_train = y_train
    return new_X_train, new_y_train
#Main()  
center_images, left_images, right_images, measurements = load_image_data()
center_images, left_images, right_images, measurements = augment_images(center_images, left_images, right_images, measurements)

X_train = np.array(center_images)
y_train = np.array(measurements)
#need to print some data stats
#X_train, y_train = augment_data(X_train, y_train)

plot_measurement_histogram(y_train)  
network_model(X_train, y_train)
