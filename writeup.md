[//]: # (Image References)

[image1]: ./images/original_steering_hist.png "Original Historgram"
[image2]: ./images/orignal_fit_historgram.png "Orignal Fit Historgram"
[image3]: ./images/left.jpg "Left View"
[image4]: ./images/center.jpg "Center View"
[image5]: ./images/right.jpg "Right View"

[image6]: ./images/steering_hist.png "Steering Historgram"

[image7]: ./images/comp_images.png "Compare Images"
[image8]: ./images/historgram_good_loss.png "Historgram Good Loss"
[image9]: ./images/nvidia.png "CNN"
[image10]: ./images/banner.png "Bannor"


![alt text][image10]
## **Overview**

This project uses Behavior Cloning to instruct a car on how to steer it's away around the track.   A Convolutional Neural Network(CNN) architecture published by [Nvidia](https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/) Autonomous Team is used as a starting point for this project.  Along with input steering data, the network is augmented to address specifics with the problem of navigating the test track.  

Link to [project code](https://github.com/hbutler97/Carnd-Behavior-Cloning/blob/master/model.py)

Link to [Result YouTube Video](https://youtu.be/caXTWlKYVHk)

## **Network Architecture**

The base CNN chosen for this project was one published by the Nvidia Autonomous Driving Team.  Architecture is shown below.  As shown below the network consist of 5 convolutional layers along with 3 fully connected layers.   A deeper explanation of the network can be found [here](https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/)

![alt text][image9]


## **Data Set Exploration**

The dataset collected consisted of images from 3 camera angles(left, center, right), and the steering angle at the time of the image capture.  Histogram of the original dataset is shown below.  

![alt text][image1]

## **Initial Results**

Initial training of the network with original data set yield a loss profile as shown below.  Adam Optimizer was used.

![alt text][image2]

As shown below, the validation loss is increasing while training loss is decreasing which is an indication of overfitting.  Additionally the car didn't making it very far without running off the track.  

## **Data Set Augmentation**

During the first test run, the car seemed to have a resistance to take turns.  This result seemed to line up well with the dataset as the majority of the dataset has a 0 steering angle.  The dataset was augmented in the following ways to show the network more of the behavior necessary to naviagte the track.

1. Zero Steering Angle reduction
* To remove the strong bias of steering straight, the dataset was augmented by removing 90% of the zero angled steering measurements images.
2. Flipping Images
* Increase the amount of training data each image was flipped on it's vertical axis
3. Adding Left and Right Cameras
* To help with directing the car back to the center of the track, the left and right camera angles were added for the remaining center. For the remaining zero steering angle images, the left and right images were added with a steering correction of 0.2 applied.

<img src="./images/left.jpg" width="260"/> <img src="./images/center.jpg" width="260"/> <img src="./images/right.jpg" width="260"/>

The augmentation of the data set yielded the new distribution shown below.  The new distribution is much closer to a Gaussian Distribution with two spikes for the new added left/right camera angles.

![alt text][image6]

Lastly, the images themselves were augmented to assist with overfitting.  The images were normalized and cropped.  The Keras Cropping2D layer also seemed to have the side affect of changing the color channel. The image below shows the comparison of the initial image and the image after the augmentation. 

![alt text][image7]

## **Network Augmentation**

The changes above didn't solve the overfitting problem.  So Dropout layers with  a %50 keep probability were added after each Fully Connected Layer. Also the Adam optimizer was modified to have a learn rate of 0.0001.  Lastly the model was trained for 5 Epochs.

The figure below shows the new validation loss.

![alt text][image8]


## **Future Areas of Exploration**
1. Image processing with additional angles
* To add more data both horizontal and vertical angles could be produced with the assoicated steering corrections.  	
2. Add Throttle and Breaking data
* Throttle and Breaking data could added to help teach the model how to modify speed




