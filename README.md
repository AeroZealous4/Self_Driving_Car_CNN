# Self_Driving_CNN
Steering Angle Control of Self Driving Car by
CNN Network and Image Processing.

The Goal of our project is to define the model which works satisfactory for
the flat road surface. We train the model and validate on the simulator.
The udacity simulator gives us the access to test our model by using the python interface.
In this simulator there are two tracks - one with flat road and another one with hilly road. We used the first track to validate our model.

Udacity Simulator: https://github.com/udacity/self-driving-car-sim

drive.py:  This file gets input frame data from simulator and feed it to trained model. Output of model is then sent to simulator for driving a car. This file is taken from                  https://github.com/ManajitPal/DeepLearningForSelfDrivingCars/blob/master/drive.py 

We first generated training data from Udacity simulator and used it for training our CNN model.

References:

(1)https://towardsdatascience.com/deep-learning-for-self-driving-cars-7f198ef4cfa2 or https://arxiv.org/pdf/1604.07316.pdf
(2)https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf
(3)https://github.com/udacity/self-driving-car-sim
