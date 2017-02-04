# Behavioral Cloning


##Description
This project use deepe learning to train a model to do the behavioral cloning of car driving. It build a CNN network with multiple layers.
The model input are images captured from the emulator cameras, and the steering wheel when driving in the emulator. The target of the model is to predict the
steering wheel angle by giving the new camera images when the car drive in different sceneries.

## Data Collection

The data collected from Udacity emulator. When select the car to training mode, you can control the car moving by mouse and keyboard.
1. 3327 records: Data collected from the Beta version Emulator use mouse as angel input
2. 8037 records: from udacity
3. 34703 records: from udacity students sharing

## Data PreProcessing

###Resize the data to 66*200*3 to adapt the Nvidia model image size
###Combine the all the data source into one input in the data generator and yield it
###Split data into train/validation/test
    1000 images for validation, 1000 images for test
###Shuffling

###RGB to YUV.
    The result from my data shows YUV didn't get better result, so I changed back to RGB.


## Model Training, Validation

The model was trained in Amazon AWS with g2.2xlarge GPU instance.

use fit_generator to generate data and save the memory

### Old Model Architecture

At first, I use convolutional neuronetwork like below, and convert the degree to categorical value, dropout as 0.5. it shows the result not good.

____________________________________________________________________________________________________
Layer (type)                     Output Shape          Param #     Connected to
====================================================================================================
convolution2d_1 (Convolution2D)  (None, 62, 196, 24)   1824        convolution2d_input_1[0][0]
____________________________________________________________________________________________________
maxpooling2d_1 (MaxPooling2D)    (None, 31, 98, 24)    0           convolution2d_1[0][0]
____________________________________________________________________________________________________
dropout_1 (Dropout)              (None, 31, 98, 24)    0           maxpooling2d_1[0][0]
____________________________________________________________________________________________________
activation_1 (Activation)        (None, 31, 98, 24)    0           dropout_1[0][0]
____________________________________________________________________________________________________
convolution2d_2 (Convolution2D)  (None, 27, 94, 36)    21636       activation_1[0][0]
____________________________________________________________________________________________________
maxpooling2d_2 (MaxPooling2D)    (None, 13, 47, 36)    0           convolution2d_2[0][0]
____________________________________________________________________________________________________
dropout_2 (Dropout)              (None, 13, 47, 36)    0           maxpooling2d_2[0][0]
____________________________________________________________________________________________________
activation_2 (Activation)        (None, 13, 47, 36)    0           dropout_2[0][0]
____________________________________________________________________________________________________
convolution2d_3 (Convolution2D)  (None, 11, 45, 48)    15600       activation_2[0][0]
____________________________________________________________________________________________________
maxpooling2d_3 (MaxPooling2D)    (None, 5, 22, 48)     0           convolution2d_3[0][0]
____________________________________________________________________________________________________
dropout_3 (Dropout)              (None, 5, 22, 48)     0           maxpooling2d_3[0][0]
____________________________________________________________________________________________________
activation_3 (Activation)        (None, 5, 22, 48)     0           dropout_3[0][0]
____________________________________________________________________________________________________
convolution2d_4 (Convolution2D)  (None, 3, 20, 64)     27712       activation_3[0][0]
____________________________________________________________________________________________________
activation_4 (Activation)        (None, 3, 20, 64)     0           convolution2d_4[0][0]
____________________________________________________________________________________________________
convolution2d_5 (Convolution2D)  (None, 1, 18, 64)     36928       activation_4[0][0]
____________________________________________________________________________________________________
activation_5 (Activation)        (None, 1, 18, 64)     0           convolution2d_5[0][0]
____________________________________________________________________________________________________
flatten_1 (Flatten)              (None, 1152)          0           activation_5[0][0]
____________________________________________________________________________________________________
dense_1 (Dense)                  (None, 110)           126830      flatten_1[0][0]
____________________________________________________________________________________________________
dense_2 (Dense)                  (None, 55)            6105        dense_1[0][0]
____________________________________________________________________________________________________
dense_3 (Dense)                  (None, 11)            616         dense_2[0][0]
====================================================================================================

### Current Model Architecture
Then I made some changes, and finally with this architecture, it gives better performance than the old one.

The model rarchitecture refer to Nvidia's paper. The model use 9 layers
###1 Normalization layer. I moved it to preprocess part
    The normalization will accelarate the computation speed by GPU.
###5 Convolutional layers.
First 3 layers use strides 2*2, and kernel 5*5. Last 2 layers use kernal 3*3 without strides.
###3 full connected layers.
3 fully connected layers with 1164,100,50,10 neurons. Finally output to steering angles.

____________________________________________________________________________________________________
Layer (type)                     Output Shape          Param #     Connected to
====================================================================================================
lambda_1 (Lambda)                (None, 66, 200, 3)    0           lambda_input_1[0][0]
____________________________________________________________________________________________________
convolution2d_1 (Convolution2D)  (None, 31, 98, 24)    1824        lambda_1[0][0]
____________________________________________________________________________________________________
elu_1 (ELU)                      (None, 31, 98, 24)    0           convolution2d_1[0][0]
____________________________________________________________________________________________________
dropout_1 (Dropout)              (None, 31, 98, 24)    0           elu_1[0][0]
____________________________________________________________________________________________________
convolution2d_2 (Convolution2D)  (None, 14, 47, 36)    21636       dropout_1[0][0]
____________________________________________________________________________________________________
elu_2 (ELU)                      (None, 14, 47, 36)    0           convolution2d_2[0][0]
____________________________________________________________________________________________________
dropout_2 (Dropout)              (None, 14, 47, 36)    0           elu_2[0][0]
____________________________________________________________________________________________________
convolution2d_3 (Convolution2D)  (None, 5, 22, 48)     43248       dropout_2[0][0]
____________________________________________________________________________________________________
elu_3 (ELU)                      (None, 5, 22, 48)     0           convolution2d_3[0][0]
____________________________________________________________________________________________________
dropout_3 (Dropout)              (None, 5, 22, 48)     0           elu_3[0][0]
____________________________________________________________________________________________________
convolution2d_4 (Convolution2D)  (None, 3, 20, 64)     27712       dropout_3[0][0]
____________________________________________________________________________________________________
elu_4 (ELU)                      (None, 3, 20, 64)     0           convolution2d_4[0][0]
____________________________________________________________________________________________________
dropout_4 (Dropout)              (None, 3, 20, 64)     0           elu_4[0][0]
____________________________________________________________________________________________________
convolution2d_5 (Convolution2D)  (None, 1, 18, 64)     36928       dropout_4[0][0]
____________________________________________________________________________________________________
elu_5 (ELU)                      (None, 1, 18, 64)     0           convolution2d_5[0][0]
____________________________________________________________________________________________________
dropout_5 (Dropout)              (None, 1, 18, 64)     0           elu_5[0][0]
____________________________________________________________________________________________________
flatten_1 (Flatten)              (None, 1152)          0           dropout_5[0][0]
____________________________________________________________________________________________________
dense_1 (Dense)                  (None, 100)           115300      flatten_1[0][0]
____________________________________________________________________________________________________
elu_6 (ELU)                      (None, 100)           0           dense_1[0][0]
____________________________________________________________________________________________________
dropout_6 (Dropout)              (None, 100)           0           elu_6[0][0]
____________________________________________________________________________________________________
dense_2 (Dense)                  (None, 50)            5050        dropout_6[0][0]
____________________________________________________________________________________________________
elu_7 (ELU)                      (None, 50)            0           dense_2[0][0]
____________________________________________________________________________________________________
dropout_7 (Dropout)              (None, 50)            0           elu_7[0][0]
____________________________________________________________________________________________________
dense_3 (Dense)                  (None, 10)            510         dropout_7[0][0]
____________________________________________________________________________________________________
elu_8 (ELU)                      (None, 10)            0           dense_3[0][0]
____________________________________________________________________________________________________
dense_4 (Dense)                  (None, 1)             11          elu_8[0][0]
====================================================================================================

In the first layers, we use 0.2 as dropout to remove overfitting issues, also use L2 Regularization.

Adam optimizer was used to optimize the model.

Parameter decision:
Epochs:

    The training epochs use 5, each epoch has 25600 inputs samples by generator.
    At first, I choose 30 epochs to run the model, then I find the loss not change too much after epoch 3, so choose a safer number 5.

Batch size:
    The batch size I choose 128. I changed the size for several times, When I use larger batch size such as 1280, I find it the running ca be out of memory. Then I change it to a smaller one, finally choose 128.

![Nvidia Architecture](./source/nvidia_architecture.png)

## Testing

Testing use 10% of the data to do the testing.
Also, pick 1 image from every 1000 images to do the prediction and manully validate the results.

## Simulate the result in autonomous mode
Modify the driver.py, add the preprocess function before send the data to the model.
Run driver.py with saved model.
Change the Emulator in AUTONOMOUS MODE and run.
The Emulator will run in autonomous mode, change the wheels according to different conditions.
![AUTONOMOUS RESULT](./source/autonomous_01.png)


## Improvements

## References
![End to End Learning for Self-Driving Cars](https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf)
![Learning a Driving Simulator](https://arxiv.org/pdf/1608.01230v1.pdf)

## TODOs
Add rotation.
Use left and right camera
Add more data to fix the cases when the car off the road


