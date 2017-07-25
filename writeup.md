[//]: # (Image References)

[image1]: ./examples/visualization.png "Visualization"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./examples/placeholder.png "Traffic Sign 1"
[image5]: ./examples/placeholder.png "Traffic Sign 2"
[image6]: ./examples/placeholder.png "Traffic Sign 3"
[image7]: ./examples/placeholder.png "Traffic Sign 4"
[image8]: ./examples/placeholder.png "Traffic Sign 5"
[image9]:  ./examples/stop0.png "Stop Sign 0"
[image10]: ./examples/stop1.png "Stop Sign 1"
[image11]: ./examples/stop2.png "Stop Sign 2"
[image12]: ./examples/stop3.png "Stop Sign 3"
[image13]: ./examples/stop4.png "Stop Sign 4"
[image14]: ./examples/stop5.png "Stop Sign 5"
[image15]: ./examples/stop6.png "Stop Sign 6"
[image16]: ./examples/stop7.png "Stop Sign 7"
[image17]: ./examples/stop8.png "Stop Sign 8"
[image18]: ./examples/stop9.png "Stop Sign 9"
[image19]: ./examples/hist.png "Data Set Historgram"
[image20]: ./examples/initial_training.png "Initial Training Results"
[image21]: ./examples/normization_training.png "Normization Training Results"
[image22]: ./examples/grey_norm_filter_training.png "Grey Norm and Filter Training Results"
[image23]: ./examples/grey_norm_filter_drop_training.png "Grey Norm and Filter Dropout Training Results"
[image24]: ./examples/grey_norm_filter_drop_training_loss.png "Grey Norm and Filter Dropout Training Results"
[image25]: ./examples/lenet5.png "LeNet CNN"
[image26]: ./examples/train_classes.png "Traning Set Histagram"

[image27]: ./examples/speed20_0.png "Speed20_ Sign 0"
[image28]: ./examples/speed20_1.png "Speed20_ Sign 1"
[image29]: ./examples/speed20_2.png "Speed20_ Sign 2"
[image30]: ./examples/speed20_3.png "Speed20_ Sign 3"
[image31]: ./examples/speed20_4.png "Speed20_ Sign 4"
[image32]: ./examples/speed20_5.png "Speed20_ Sign 5"
[image33]: ./examples/speed20_6.png "Speed20_ Sign 6"
[image34]: ./examples/speed20_7.png "Speed20_ Sign 7"
[image35]: ./examples/speed20_8.png "Speed20_ Sign 8"
[image36]: ./examples/speed20_9.png "Speed20_ Sign 9"

[image37]: ./examples/processed_stop_report.png "Processed Stop Sign"
[image38]: ./examples/stop_report.png "Stop Sign Report"

[image40]: ./examples/speed30_0.png "Speed30_ Sign 0"
[image41]: ./examples/speed30_1.png "Speed30_ Sign 1"
[image42]: ./examples/speed30_2.png "Speed30_ Sign 2"
[image43]: ./examples/speed30_3.png "Speed30_ Sign 3"
[image44]: ./examples/speed30_4.png "Speed30_ Sign 4"
[image45]: ./examples/speed30_5.png "Speed30_ Sign 5"
[image46]: ./examples/speed30_6.png "Speed30_ Sign 6"
[image47]: ./examples/speed30_7.png "Speed30_ Sign 7"
[image48]: ./examples/speed30_8.png "Speed30_ Sign 8"
[image49]: ./examples/speed30_9.png "Speed30_ Sign 9"

[image50]: ./examples/child0.png "Child_ Sign 0"
[image51]: ./examples/child1.png "Child_ Sign 1"
[image52]: ./examples/child2.png "Child_ Sign 2"
[image53]: ./examples/child3.png "Child_ Sign 3"
[image54]: ./examples/child4.png "Child_ Sign 4"
[image55]: ./examples/child5.png "Child_ Sign 5"
[image56]: ./examples/child6.png "Child_ Sign 6"
[image57]: ./examples/child7.png "Child_ Sign 7"
[image58]: ./examples/child8.png "Child_ Sign 8"
[image59]: ./examples/child9.png "Child_ Sign 9"

[image60]: ./german_signs/yield.png "Processed German Sign"
[image61]: ./german_signs/yield_color.png "Raw German Sign"
[image62]: ./german_signs/german_results0.png "Results"
[image63]: ./german_signs/german_results1.png "Results"
[image64]: ./german_signs/german_results2.png "Results"
[image65]: ./german_signs/german_results3.png "Results"
[image66]: ./german_signs/german_results4.png "Results"
[image67]: ./german_signs/german_results5.png "Results"
[image68]: ./german_signs/german_results6.png "Results"
[image69]: ./german_signs/german_results7.png "Results"


# **Traffic Sign Recognition** 

## **Overview**

This project classifies German Traffic Signs using a Convolutional Neural Network(CNN).   [LeNet CNN](http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf) is used as a starting point for this project.  The architecture for the [LeNet CNN](http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf) is shown below.  Along with LeNet architecture augmentation and training, the German Traffic Sign data set is preprocessed to achieve a target classification accuracy of 93%.

Link to [project code](https://github.com/hbutler97/Traffic-Sign-Classifier/blob/master/Traffic_Sign_Classifier.ipynb)

![alt text][image25]

## **Data Set Exploration**

The German Traffic Sign data set consisted of a training set, validation set and test set. 

#### **Data Set Summary**

* The size of training set is: **34799**
* The size of the validation set is: **4410**
* The size of test set is: **12630**
* The shape of a traffic sign image is: **(32, 32, 3)**
* The number of unique classes/labels in the data set is: **43**

#### **Data Set Distribution**

Each image is also classified in one of 43 classes.  

![alt text][image26]

#### **Data Set Visualization**

10 random images from each classification was visualized in order to view the integrity of the data set.
As seen below, images have varying quality levels and attributes which increases the features that the CNN will have to understand to properly classify the traffic signs.


![alt text][image9] ![alt text][image10] ![alt text][image11] ![alt text][image12] ![alt text][image13] ![alt text][image14] ![alt text][image15] ![alt text][image16] ![alt text][image17] ![alt text][image18]

![alt text][image27] ![alt text][image28] ![alt text][image29] ![alt text][image30] ![alt text][image31] ![alt text][image32] ![alt text][image33] ![alt text][image34] ![alt text][image35] ![alt text][image36]

![alt text][image40] ![alt text][image41] ![alt text][image42] ![alt text][image43] ![alt text][image44] ![alt text][image45] ![alt text][image46] ![alt text][image47] ![alt text][image48] ![alt text][image49]

![alt text][image50] ![alt text][image51] ![alt text][image52] ![alt text][image53] ![alt text][image54] ![alt text][image55] ![alt text][image56] ![alt text][image57] ![alt text][image58] ![alt text][image59]


**Initial LeNet Training results**

* Number of EPOCH 10
* Batch size: 128
* Learn Rate: 0.001

![alt text][image20]

* Training Accuracy: 97.6%
* Validation Accuracy: 87.2%

The high training accuracy and low validation accuracy implies that the model is overfitting. The following was used to address Overfitting:

* **Data Preprocessing**

* **Drop Out**

* **L2 Regularization** 


#### **Data Set Preprocessing**

As stated above, the data set is complex and as such contributes to the overfitting of the network.  Images were converted to greyscale, passed through a [Contrast Limited Adaptive Histogram Equalization(CLAHE)](https://en.wikipedia.org/wiki/Adaptive_histogram_equalization) Filter and normalized to simplify the image feature set.  Below is an example of an image before and after the transformation.


![alt text][image38] ![alt text][image37]


**LeNet Training results after image preprocessing**

* Number of EPOCH 10
* Batch size: 128
* Learn Rate: 0.001

![alt text][image22]

* Training Accuracy: 99.5%
* Validation Accuracy: 92.6%

Preprocessing gets the Validation Accuracy very close to the test data target of 93%.


#### **Design and Test Model Architecture**

The architecture of the Model used is shown below...  The augmentation to the Default LeNet architecture where adding dropout between fully connected nets and adding L2 Regularization.  Both shown below in bold.
| Layer         	|     Description	        		| 
|:---------------------:|:---------------------------------------------:| 
| Input         	| 32x32x1 Gray Scale image   			| 
| 1. Convolution 5x5   	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU			|						|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6  		|
| 2. Convolution 5x5   	| 1x1 stride, valid padding, outputs 10x10x16 	|
| RELU			|						|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16  		        |
| Flatten               | output 400					|
| 3. Fully Connected    | output 120					|
| RELU			|						|
| **Dropout**		| keep prob 0.5					|
| 4. Fully Connected    | output 84					|
| RELU			|						|
| **Dropout**		| keep prob 0.5					|
| 5. Fully Connected	| output 43 					|
| Softmax		|         					|
| **L2 Regularizer**	| beta 0.01					|



**Augmented LeNet Training results**

* Number of EPOCH 100
* Batch size: 128
* Learn Rate: 0.0007
* Regularization Loss Constant: 0.01

![alt text][image23] 

* Training Accuracy: 100.0%
* Validation Accuracy: 97.6%

With the changes to the model, training and validation accuracy has improved as shown in the figure above

Figure below is a graph of the loss function which also includes the L2 loss(Artificial constrain)

![alt text][image24]


**Augmented LeNet Test results**

* <span style="color:red"> Test Accuracy: 95% </span>



#### **Test Model on New Images**

8 Random images from the web were downloaded and applied to the model..  Prior to running the images through the model, the dimensions of the image were reduced to fit the network which causes some loss. 


Raw Images

![alt text][image61]

Processed Images

![alt text][image60]

The sized reduced image was then preprocess and applied to the model.



Image 1 Yield Accuracy = 1.000

Image 2 50KM/Hr Accuracy = 1.000

Image 3 Road Work Accuracy = 1.000

Image 4 Don't Enter Accuracy = 1.000

Image 5 30KM/Hr Accuracy = 1.000

Image 6 60KH/Hr Accuracy = 0.833

Possible issues are the serif on the "6"

Image 7 Don't Pass Accuracy = 0.714

Possible issues are the water marks and pole

Image 8 Don't Pass 2 Accuracy = 0.625

Possible issues are the water marks angle of the shot and pole


Yield
![alt text][image62]
50 KM/hr
![alt text][image63]
Road Work
![alt text][image64]
Don't Enter
![alt text][image65]
30 KM/hr
![alt text][image66]
60 KM/hr
![alt text][image67]
Don't Pass
![alt text][image68]
Don't Pass 2
![alt text][image69]

The model properly predicts the first 5 signs with a very high degree of certainty of it's selection.  On the later three the image it guesses with the highest probability is difficult to classify visually by human eyes. which is interesting. This may be an issue with the similarities of the images in the sets for these classes. 



