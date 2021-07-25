# **Behavioral Cloning Project** 

---
[//]: # (Image References)

[image1]: ./Plots/overall.png "Overall"
[image2]: ./Plots/cnn.png "NVIDIA CNN Model"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"


**Behavioral Cloning Project For Self Driving Car**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


![alt text][image1]

The above image shows a general block diagram (NVIDIA project) of the self driving car steering method through behavior cloning. As it can be seen the first step is to collect data through a user of the simulator. This data includes three dashcamera images pointing (left, centre and right), moreover the control inputs to the car are also included, such as steering angle and throttle. For the case of driving the car autonomously through the tracks, the throttle is ignored. 

Convolutional Neural Network (CNN) is used in saveral layers of the training model. The process of developing the model includes making use of keras API, in which it takes the three images as well as the control steering angle as an input. Saveral recorded scenarios were included such as driving in the opposite direction of the track to create a more general solution. The model is stored in 'model.h5'


---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy
#### NVIDIA Model

![alt text][image2]

The above shows the NVIDIA AI model that was used for the self driving car project above. as it can be seen the model uses five CNN layers on top of each other. This is done to observe and gather more details from the training images. Moreover, each layer represent a level of detail, as the layers goes smaller , finer details are detected and recorded, and the weights are adjusted accordingly. Normalization laywer is following the first input layer, this is done to make it easier for the model to process the images. 

#### 1. An appropriate model architecture has been employed

My model consists of five convolution neural network (similar to Nvidia model) with 3x3 filter sizes and  strides 2X2. 

The model includes RELU layers to introduce nonlinearity, and the data is normalized in the model using a Keras lambda layer. Moreover, Max pooling is used to reduce the size of the image features, hence reduce the number of training paramters in the model. Two dropout layers are used to reduce and prevent overfitting of the model. A densely connected layer is used to provide learning features from all the combinations of the features of the previous layer. Cropping of the images is used to ignore the irrelevent off road features, such as the sky, mountains, trees... etc. 

A summary of the used model are shown below. 

```
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
lambda (Lambda)              (None, 160, 320, 3)       0         
_________________________________________________________________
cropping2d (Cropping2D)      (None, 75, 320, 3)        0         
_________________________________________________________________
conv2d (Conv2D)              (None, 36, 158, 24)       1824      
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 16, 77, 36)        21636     
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 6, 37, 48)         43248     
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 4, 35, 64)         27712     
_________________________________________________________________
conv2d_4 (Conv2D)            (None, 2, 33, 64)         36928     
_________________________________________________________________
max_pooling2d (MaxPooling2D) (None, 1, 16, 64)         0         
_________________________________________________________________
flatten (Flatten)            (None, 1024)              0         
_________________________________________________________________
dense (Dense)                (None, 100)               102500    
_________________________________________________________________
dropout (Dropout)            (None, 100)               0         
_________________________________________________________________
dense_1 (Dense)              (None, 50)                5050      
_________________________________________________________________
dense_2 (Dense)              (None, 10)                510       
_________________________________________________________________
dropout_1 (Dropout)          (None, 10)                0         
_________________________________________________________________
dense_3 (Dense)              (None, 1)                 11        
=================================================================
Total params: 239,419
Trainable params: 239,419
Non-trainable params: 0
_____________________________________________
```
#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 21). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 10-16). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 25).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road ... 

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to ...

My first step was to use a convolution neural network model similar to the ... I thought this model might be appropriate because ...

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model so that ...

Then I ... 

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track... to improve the driving behavior in these cases, I ....

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 18-24) consisted of a convolution neural network with the following layers and layer sizes ...

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)


#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to .... These images show what a recovery looks like starting from ... :

![alt text][image3]
![alt text][image4]
![alt text][image5]

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles thinking that this would ... For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]

Etc ....

After the collection process, I had X number of data points. I then preprocessed this data by ...


I finally randomly shuffled the data set and put Y% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was Z as evidenced by ... I used an adam optimizer so that manually training the learning rate wasn't necessary.
