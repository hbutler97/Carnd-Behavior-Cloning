import csv
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

from random import shuffle

def compare_images(left_image, right_image):
    print(type(left_image))
    print(left_image.shape)
    print(type(right_image))
    print(right_image.shape)
    
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
    f.tight_layout()
    #ax1.imshow(cv2.cvtColor(left_image, cv2.COLOR_BGR2RGB))
    ax1.imshow(left_image)
    ax1.set_title('Shape '+ str(left_image.shape),
                  fontsize=50)
    #ax2.imshow(cv2.cvtColor(right_image, cv2.COLOR_BGR2RGB))
    ax2.imshow(right_image)
    #cv2.imshow('image', right_image)
    ax2.set_title('Shape '+ str(right_image.shape)
                  , fontsize=50)
    plt.show()

def compare_cv_images(left_image, right_image):

    resized_image = cv2.resize(right_image, (320, 160))
    print(left_image.shape)
    print(resized_image.shape)
    
    final_frame = cv2.hconcat((left_image, resized_image))
    cv2.imshow('lena', final_frame)
    #cv2.imshow('image',left_image)
    #cv2.imshow('image', right_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
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
    #center_images, measurements = reduce_centered_side_images(0.5, center_images, measurements, steering_correction)
    #center_images, measurements = reduce_centered_side_images(0.5, center_images, measurements, -steering_correction)
 
    
    return center_images, left_images, right_images, measurements

def plot_loss_histogram(history_object):  
    plt.plot(history_object.history['loss'])
    plt.plot(history_object.history['val_loss'])
    plt.title('model mean squared error loss')
    plt.ylabel('mean squared error loss')
    plt.xlabel('epoch')
    plt.legend(['training set', 'validation set'], loc='upper right')
    plt.savefig('images/historgram.png')
    
def plot_measurement_histogram(measurements):  
    plt.hist(measurements, bins=100)
    plt.title('Steering Angle Historgram')
    plt.ylabel('number of measurements')
    plt.xlabel('angle')
    plt.show()
    #plt.savefig('images/steering_hist.png')


from keras.models import Sequential, Model, load_model
from keras import backend as K
import theano
from keras.layers import Flatten, Dense, Lambda, Dropout
from keras.layers.convolutional import Convolution2D, Cropping2D
from keras.layers.pooling import MaxPooling2D
from keras.optimizers import Adam

def visualize_model(X_train, models):
    model = load_model('model.h5')
    
    crop_layer = model.layers[2].output
    output_fn = K.function([model.layers[0].input], [crop_layer])

    input_image = X_train[0:1:,:,:]
    print(input_image.shape)
    

    output_image = output_fn([input_image])
    print(output_image[0].shape)

    output_image = np.rollaxis(np.rollaxis(output_image[0], 3, 1), 3, 1)
    print(output_image.shape)



    print(output_image.shape)

    fig=plt.figure(figsize=(8,8))
    for i  in range(31):
        ax = fig.add_subplot(6,6,i+1)
        ax.imshow(output_image[0,:,:,i], cmap=matplotlib.cm.gray)
        plt.xticks(np.array([]))
        plt.yticks(np.array([]))
        plt.tight_layout()
    plt.show()
#    compare_images(input_image[0,:,:,:],output_image[0][0,:,:,:])
    exit()

    
    

def network_model(X_train, y_train, epochs):
    
    model = Sequential()
    #add normalization
    model.add(Lambda(lambda x: x /255.0 - 0.5 , input_shape=(160, 320, 3)))
    model.add(Cropping2D(cropping=((70,25),(0,0))))
    # Conv Layer #0 (depth=3, kernel=1x1) - change color space
    #model.add(Convolution2D(3, 1, 1, border_mode='same'))
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

    #adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.01)
    model.compile(loss='mse', optimizer='adam')
    #model.compile(loss='mse', optimizer=adam)
    history_object = model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=epochs)
    model.save('model.h5')
#    plot_loss_histogram(history_object)
    return model
def augment_images(X_train, y_train):
    new_X_train = X_train
    new_y_train = y_train
    return new_X_train, new_y_train
#Main()  
center_images, left_images, right_images, measurements = load_image_data()
center_images, left_images, right_images, measurements = augment_data(center_images, left_images, right_images, measurements)

X_train = np.array(center_images)
y_train = np.array(measurements)
plot_measurement_histogram(y_train)

model = network_model(X_train, y_train,10)
#visualize_model(X_train,model)




