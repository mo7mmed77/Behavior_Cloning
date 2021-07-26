# **Behavioral Cloning Project** 

---
[//]: # (Image References)

[image1]: ./Plots/overall.png "Overall"
[image2]: ./Plots/cnn.png "NVIDIA CNN Model"
[image3]: ./Plots/Image_Processed.png "Cropped Image"
[image4]: ./Plots/sim2.png "Lane"
[image5]: ./Plots/Model_MSE_Loss.png "Recovery Image"
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

My model mainly consists of five convolution neural network (similar to Nvidia model) with strides 2X2. The input layer is used similar to the expected input from the simulator (160, 320, 3). 

The model includes RELU layers to introduce nonlinearity, and the data is normalized in the model using a Keras lambda layer. Moreover, Max pooling is used to reduce the size of the image features, hence reduce the number of training paramters in the model. Two dropout layers are used to reduce and prevent overfitting of the model. A densely connected layer is used to provide learning features from all the combinations of the features of the previous layer. Cropping of the images is used to ignore the irrelevent off road features, such as the sky, mountains, trees... etc. 

A summary of the used model are shown below. 

```
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
lambda (Lambda)              (None, 160, 320, 3)       0         
_________________________________________________________________
dropout (Dropout)            (None, 160, 320, 3)       0         
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
_________________________________________________________________
```

#### Data Used in Training 
The data used in training are summarized in the following table 
| Data Folder Name     | Description |  Number of samples |
| ----------- | ----------- |----------- |
| data      | This is the data provided by Udacity at their own github repository       |  24,108  |
| mydata   | This my training data of three laps        | 23,556  |
| mydata_oppos   |  two laps driving in opposite direction       |  8,142  |
| mydata_avoid_dirt   |  six times driving and avoiding the offroad turn       | 2,745   |

The total number of samples is 58,551. The sample includes images of the cameras on the car pointing centre, left and right.  This number is doubled by flipping all images and steering input, so the final training data is 117,102. 

###### Steering Correction factor
A correction factor is introduced to the left and right images (0.1 deg). 0.2 factor was tested and the vehicle was always steering to the right of the lane. 

##### Image processing 
The images are cropped to eliminate the irrelevent features such as the mountains, sky and trees. Moreover, it is cropped from the bottom to eliminate the hood features. 
```
    model.add(Cropping2D(cropping=((60,25),(0,0)))) 
```

![alt text][image3]


#### Attempts to reduce overfitting in the model

The model contains a dropout layer in order to reduce overfitting. The output of this layer are randomly subsampled, it also has an effect of thinning the network while training. The dropout rate was chosen to be 50% before reaching the output layer, and 80% after the input, this was chosen according to MachineLearningMyster.com [Link](https://machinelearningmastery.com/dropout-for-regularizing-deep-neural-networks). 


```
 model.add(Dropout(0.5))
 
 ```


Moreover, as it can be seen earlier a data of driving the track in the opposite direction was used. This should also contribute to reducing overfitting of the model. 

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was 0.001. I had some issues providing a good model, then i tested with values 0.0001 but the model was giving almost the same results then i went back and stuck with this number. 

The Validation set was  tested with 20% and 25%, the 20% was providing lower values of the mean squared error (MSE). Hence 20% was chosen. Two dropout layers was chosen and tuned to be 0.8 and 0.5 for the input and output layer, respectively.  Other than that the model was chosen similar to the Nvidia model.

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used the given data and I added three laps (mydata) of driving at the centre lane and during each lap I would let the car get closer to the left or right lane then correct it. This is to ensure that the car will correct itself in Autonomous mode, because it incounters similar scenario. Moreover, the car had issues correcting the lane during this left turn. 
![alt text][image4]

In which there is an offroad exit and the right lane line is missing. I decreased this issue by providing more data during this left turn only. I did this exact turn 6  times and stored it under (mydata_avoid_dirt). The model then behaved much better. Furthermore, as mentioned earlier another two laps were gathered from driving in the opposition direction. 


### Final Model Architecture and Training Strategy Remarks

My first step was to use many layers of convolution neural network model similar to the Nvidia model I thought this model might be appropriate because it solved similar problem. 

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model so that it includes dropout layers. 

Then I was able to reduce the MSE for the validation set.  

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track as mentioned earlier during the left turn where the right lane is missing. Hence to improve the driving behavior in these cases, I included more data during that turn. 

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

I finally randomly shuffled the data set 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 7 as evidenced by experimentation. This number was chosen based on the output MSE reaching a steady state value.  I used an adam optimizer so that manually training the learning rate wasn't necessary.


#### Model MSE output
![alt text][image3]

as it can be seen the MSE for test set is very low and decreasing with each epoch, which means that the model should be accurate. The validation set are not decreasing , this could be due to the high number of data used during training (117,102 samples), However, this value is small enough that the model should generalize well. 

#### Video of the Run 
(Video.mp4) is recorded during autonomous mode. It summarizes the performance of the model of more than one lap of the track. Another video (run2.mp4) is recorded as well to guarentee that the model runs well. 
#### Summary of Issues 
* The GPU sometimes are not identified when starting the model for training. This was sometimes solved by restarting the environment. 
* The memory is sometimes too low for the training of the model. This was solved by closing any other applications that takes too much memory. 
* The car will drive into the offroad exit. This was solved by adding more data during that turn. 
* Sometimes during autonomous mode, the input steering to the car would freeze at a certain angle, which means that even the (drive.py) output data to the simulator are freezed. This is simply solved by restarting the run of (drive.py) and the simulator. 
