# **Behavioral Cloning** 

---

**Behavrioal Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[left]: ./left.jpg "left camera"
[center]: ./center.jpg "center camera"
[right]: ./right.jpg "right camera"
[recover_left]: ./recover_left.jpg "Recovery Image1"
[recover_right]: ./recover_right.jpg "Recovery Image2"
[crop]: ./crop.jpg "cropped Image"
[flip]: ./im_flip.jpg "Flipped Image"

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* preprocess.ipynb preprocess the data in junyper notebook
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing the weight of trained network 
* model.json containing the model
* writeup_report.md summarizing the results

#### 2. Submssion includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py
```

#### 3. Submssion code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model arcthiecture has been employed

I'm using Nvidia model which introduced in this paper:(http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf). 

The first layer is a normalization layer.

Following are 5 convolution layers, and 3 fully connected layers.

The model includes RELU activation function in each layer to introduce nonlinearity.

#### 2. Attempts to reduce overfitting in the model

The model contains a dropout layer in order to reduce overfitting. 

The model was trained and validated on different data sets to ensure that the model was not overfitting. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually.

#### 4. Appropriate training data

I'm using Udacity data. I also used PS4 controller run 2 laps to generate additional data, incluidng recovery data, but in the final model I just used Udacity data and the result is good.

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The approach I used can be described in two steps: 

1.Data exploration and preprocessing 
2.Model design and tuning

First I did some data exploration then some preprocess, since the zero angle data is dominating, I screen out 80% zero angle data, and for image from left camera I add +0.25 to the angle, -0.25 to the angle for right camera image. I crop the sky and the hood of the car out, since they are irrelavent information. And I mirrored all image to balance left turn and right turn.

Then I tried feed the data to the model, first I tried with the model from comma.ai, but the car keep turning left, then I add 2 more dense layers, but wasn't able to improve the model. Then I tried to use Udacity data only, this time the car was able to drive within the lane until the first turn. Then I switched to Nvidia model which contains more convolution layers, and the car was able to drive in the lane.

#### 2. Final Model Architecture

The network consists of 9 layers, including a normalization layer, 5 convolutional layers and 3 fully connected layers. 

The first layer is a normalization layer, which normalize image values between -0.5 to 0.5.

The first 3 convolution layers are using 2x2 stride and 5x5 kernel, the last 2 convolution layers are non-stride with 3x3 kernel size.

Following are 3 fully connected layers leading to an output control value which is the steering angle.

#### 3. Creation of the Training Set & Training Process

I'm using Udacity data set to train the model, though I record my own data and appended it to Udacity data at first, it turns out using Udacity data only has better result.

Here is what I do to record my own data:

I first recorded two laps on track one using stable simulator, which mean I have center/left/right data. Here is example images:

![alt text][center]
![alt text][left]
![alt text][right]

I then recorded the car recovering from the left side and right sides of the road back to center so that the car would learn to recover. Here are two images of the car recovering from left/right:

![alt text][recover_left]
![alt text][recover_right]

To augment the data sat, I crop the image to help increase the model training speed and accuracy(turns out this helps a lot!), and also flipped images and angles to balance the left turn and right turn. 

![alt text][crop]
![alt text][flip]


After the collection process, I had 43980 number of data points. But later I screen my own data out and only use Udacity data.

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. At first I used 10 epochs, and I found the loss doesn't improve much since 5 epochs, so I use 5 epochs in the final model. I used an adam optimizer so that manually training the learning rate wasn't necessary.
