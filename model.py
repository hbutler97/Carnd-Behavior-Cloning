import csv
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from random import shuffle
#
def bgr_to_rgb(image):
    b,g,r = cv2.split(image)
    return cv2.merge([r,g,b])

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

        image = bgr_to_rgb(cv2.imread(current_path))
        images.append(image)

        left_image = bgr_to_rgb(cv2.imread(left_current_path))

        left_images.append(image)
        right_image = bgr_to_rgb(cv2.imread(right_current_path))

        right_images.append(image)
        
        measurement = float(line[3])
        measurements.append(measurement)

    return images, left_images, right_images, measurements

#Function used to modify the number of images that are the right left images off center
def reduce_centered_side_images(reduce_percent, center_images, measurements,steering_angle):
    
    temp_center_images = []
    temp_measurements = []

    original_size = len(measurements)
    numb_zero_measurements = measurements.count(0.0)
    numb_to_remove = int(numb_zero_measurements * reduce_percent)
    
    print("Removing %d of %d centered and side image data\n" %(numb_to_remove, numb_zero_measurements))

    for index in range(original_size):
        if measurements[index] != steering_angle:
            temp_center_images.append(center_images[index])
            temp_measurements.append(measurements[index])
        else:
            numb_to_remove = numb_to_remove - 1
            if numb_to_remove <= 0:
                temp_center_images.append(center_images[index])
                temp_measurements.append(measurements[index])

    return temp_center_images, temp_measurements

#Function used to modify the number of zero steering angle images
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
def add_side_cameras(steer_adjust, reduce_center_percent, center_images, left_images, right_images, measurements):
    temp_center_images = []
    temp_measurements = []
    
    numb_zero_measurements = measurements.count(0.0)
    numb_to_remove = int(numb_zero_measurements * reduce_center_percent)
    
    print("Adding Side Cameras\n")
    
    for index, measurement in enumerate(measurements):
        if measurements[index] != 0.0:
            temp_center_images.append(center_images[index])
            temp_measurements.append(measurements[index])
            temp_center_images.append(left_images[index])
            temp_measurements.append(measurements[index]+steer_adjust)
            temp_center_images.append(right_images[index])
            temp_measurements.append(measurements[index]-steer_adjust)
        else:
            numb_to_remove = numb_to_remove - 1
            temp_center_images.append(center_images[index])
            temp_measurements.append(measurements[index])
            if numb_to_remove <= 0:
                temp_center_images.append(left_images[index])
                temp_measurements.append(measurements[index]+steer_adjust)
                temp_center_images.append(right_images[index])
                temp_measurements.append(measurements[index]-steer_adjust)

    print("Pre Side Carmera data size: %d Post Side Carmera data size: %d\n" %(len(measurements), len(temp_measurements)))
    return temp_center_images, temp_measurements

def flip_images(center_images, left_images, right_images,  measurements):
    temp_center_images = []
    temp_right_images = []
    temp_left_images = []
    temp_measurements = []
    
    print("Flipping Images\n")

    for index in range(len(measurements)):
        temp_center_images.append(center_images[index])
        temp_right_images.append(right_images[index])
        temp_left_images.append(left_images[index])
        temp_measurements.append(measurements[index])
        temp_center_images.append(cv2.flip(center_images[index],1))
        temp_right_images.append(cv2.flip(right_images[index],1))
        temp_left_images.append(cv2.flip(left_images[index],1))
        temp_measurements.append(measurements[index]*-1.0)
        
    print("Pre Flip data size: %d Post Flip data size: %d\n" %(len(measurements), len(temp_measurements)))
    return temp_center_images, temp_left_images, temp_right_images, temp_measurements

def augment_data(center_images, left_images, right_images, measurements):
    steering_correction = 0.2
    center_images, left_images, right_images, measurements = reduce_centered_images(0.9, center_images, left_images, right_images, measurements)
    center_images,left_images, right_images,  measurements = flip_images(center_images, left_images, right_images,  measurements)   
    center_images, measurements = add_side_cameras(steering_correction, 0.2, center_images, left_images, right_images, measurements)
    center_images, measurements = reduce_centered_side_images(0.9, center_images, measurements, 0.0)
    
    return center_images, left_images, right_images, measurements

def plot_loss_histogram(history_object):  
    plt.plot(history_object.history['loss'])
    plt.plot(history_object.history['val_loss'])
    plt.title('model mean squared error loss')
    plt.ylabel('mean squared error loss')
    plt.xlabel('epoch')
    plt.legend(['training set', 'validation set'], loc='upper right')
    plt.show()
    plt.savefig('images/historgram.png')
    
def plot_measurement_histogram(measurements):  
    plt.hist(measurements, bins=100)
    plt.title('Steering Angle Historgram')
    plt.ylabel('number of measurements')
    plt.xlabel('angle')
    plt.show()
    plt.savefig('images/steering_hist.png')


from keras.models import Sequential, Model, load_model
from keras import backend as K
import theano
from keras.layers import Flatten, Dense, Lambda, Dropout
from keras.layers.convolutional import Convolution2D, Cropping2D
from keras.layers.pooling import MaxPooling2D
from keras.optimizers import Adam

def network_model(X_train, y_train, epochs):
    
    model = Sequential()
    model.add(Lambda(lambda x: x /255.0 - 0.5 , input_shape=(160, 320, 3)))
    model.add(Cropping2D(cropping=((70,25),(0,0))))
    model.add(Convolution2D(24,5,5, subsample=(2,2), activation="relu"))
    model.add(Convolution2D(36,5,5, subsample=(2,2), activation="relu"))
    model.add(Convolution2D(48,3,3, subsample=(2,2), activation="relu"))
    model.add(Convolution2D(64,3,3, activation="relu"))
    model.add(Convolution2D(64,3,3, activation="relu"))
    model.add(Flatten())
    model.add(Dense(100))
    model.add(Dropout(0.5))
    model.add(Dense(50))
    model.add(Dropout(0.5))
    model.add(Dense(10))
    model.add(Dropout(0.5))
    model.add(Dense(1))


    #    model.compile(loss='mse', optimizer='adam',  metrics=['accuracy'])
    model.compile(loss='mse', optimizer=Adam(lr=0.0001),  metrics=['accuracy'])
    history_object = model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=epochs)
    model.save('model.h5')
    plot_loss_histogram(history_object)
    return model


#Main()  

center_images, left_images, right_images, measurements = load_image_data()
center_images, left_images, right_images, measurements = augment_data(center_images, left_images, right_images, measurements)

X_train = np.array(center_images)
y_train = np.array(measurements)

print('Train:', X_train.shape, y_train.shape)
X_train, X_train_test, y_train, y_train_test = train_test_split(X_train, y_train,test_size=0.05, random_state=42)
print('Train:', X_train.shape, y_train.shape)
print('Test:', X_train_test.shape, y_train_test.shape)



plot_measurement_histogram(y_train)
model = network_model(X_train, y_train,5)


# evaluate the model
scores = model.evaluate(X_train_test, y_train_test)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))



