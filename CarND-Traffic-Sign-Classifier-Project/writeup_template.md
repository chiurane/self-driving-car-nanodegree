# **Traffic Sign Recognition** 

## Writeup

### This is the write-up for the Traffic Sign Classifier Project.

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: https://github.com/chiurane/self-driving-car-nanodegree/blob/main/CarND-Traffic-Sign-Classifier-Project/data/visualization.PNG "Visualization"
[image2]: https://github.com/chiurane/self-driving-car-nanodegree/blob/main/CarND-Traffic-Sign-Classifier-Project/data/after_augmentation.PNG "Distribution after augmentation"
[image3]: https://github.com/chiurane/self-driving-car-nanodegree/blob/main/CarND-Traffic-Sign-Classifier-Project/data/augmented.PNG "Scaled, Translated or Rotated"
[image4]: https://github.com/chiurane/self-driving-car-nanodegree/blob/main/CarND-Traffic-Sign-Classifier-Project/data/normalize_grayscale.PNG "Normalized and Gray Scaled Image"
[image5]: https://github.com/chiurane/self-driving-car-nanodegree/blob/main/CarND-Traffic-Sign-Classifier-Project/data/model_architecture.PNG "Model Architecture"
[image6]: https://github.com/chiurane/self-driving-car-nanodegree/blob/main/CarND-Traffic-Sign-Classifier-Project/data/children_crossing.jfif "Traffic Sign 1"
[image7]: https://github.com/chiurane/self-driving-car-nanodegree/blob/main/CarND-Traffic-Sign-Classifier-Project/data/construction.jfif "Traffic Sign 2"
[image8]: https://github.com/chiurane/self-driving-car-nanodegree/blob/main/CarND-Traffic-Sign-Classifier-Project/data/go_right.jfif "Traffic Sign 3"
[image9]: https://github.com/chiurane/self-driving-car-nanodegree/blob/main/CarND-Traffic-Sign-Classifier-Project/data/go_straight_or_left.jfif "Traffic Sign 4"
[image10]: https://github.com/chiurane/self-driving-car-nanodegree/blob/main/CarND-Traffic-Sign-Classifier-Project/data/left_turn.jfif "Traffic Sign 5"
[image11]: https://github.com/chiurane/self-driving-car-nanodegree/blob/main/CarND-Traffic-Sign-Classifier-Project/data/speed_limit20.jfif "Traffic Sign 6"
[image12]: https://github.com/chiurane/self-driving-car-nanodegree/blob/main/CarND-Traffic-Sign-Classifier-Project/data/stop.jfif "Traffic Sign 7"
[image13]: https://github.com/chiurane/self-driving-car-nanodegree/blob/main/CarND-Traffic-Sign-Classifier-Project/data/straight_or_right.jfif "Traffic Sign 8"
[image14]: https://github.com/chiurane/self-driving-car-nanodegree/blob/main/CarND-Traffic-Sign-Classifier-Project/data/traffic_cycle.jfif "Traffic Sign 9"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is: 34799
* The size of the validation set is: 4410 
* The size of test set is: 12630
* The shape of a traffic sign image is: (32, 32, 3)
* The number of unique classes/labels in the data set is: 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. Here I have three bar charts showing the 
the distribution of images by class in each of the datasets:

![alt text][image1]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to convert the images to grayscale and normalize them because deep learning models
perform better and learn faster with gray scale data. 

Here is an example of a traffic sign image before and after grayscaling and normalization.

![alt text][image4]

I decided to generate additional data because the data was very unbalanced.

To add more data to the the data set, I performed three kinds of transformations
on the original image randomly. I either scaled the scaled, translated or rotated an image.

Here is an example of an original image going through augmentation:

![alt text][image3]

The distribution of data before augmentation is:
![alt text][image1]
The distribution of data after augmentation is: 
![alt text][image2]
The new training example size after augmentation is: 104450
Augmentation was only performed on training data.

#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model, which is LeNet, consisted of the following layers:
![alt text][image5]

#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used I wrote a python method called
train which has a callback and calls model.fit on my model
with the training and validation data.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of: 97.45%
* validation set accuracy of: 88.78%

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are nine German traffic signs that I found on the web:

![alt text][image6] ![alt text][image7]
![alt text][image8] ![alt text][image9] ![alt text][image10]
![alt text][image11] ![alt text][image12] ![alt text][image13]
![alt text][image14]

The first image might be difficult to classify because when in its 
grayscale it might also look like a person on bicycle.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Children Crossing		| Children Crossing   									| 
| Construction			| Road Narrows on the left 										|
| Turn Right			| Turn Right						|
| Turn Left or Straight	| Straight or Left					 				|
| Turn Left 			| Turn Left
| Speed Limit 20kmh     | Speed Limit 20kmh
| Stop                  | Speed Limit 50kmh
| Turn Right or Straight| Straight or Right
| Round about           | Round about


The model was able to correctly guess 7 of the 9 traffic signs, which gives an accuracy of 78%. This compares favorably to the accuracy on the test set of 97%

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a children crossing sign at 97%. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .97         			| Children Crossing   							| 
| .02     				| Road Narrows on the Right     				|
| <0					| Slippery Road									|
| <0         			| Dangerous Curve to the left	 				|
| <0    			    | Bumpy Road          							|


For the second image, the model is very confident that its road narrows sign but its wrong. Its actually road works which is second
| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .82        			| Road Narrows on the Right 					| 
| .17     				| Road Work 									|
| <0					| Chidren Crossing								|
| <0	      			| Double Curve  				 				|
| <0				    | Speed Limit 20km/h  							|

For the third image, the model is 100% sure that the image is Turn Right ahead and its correct
| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 99.9         			| Turn Right Ahead   							| 
| <0     				| Go Straight or Right							|
| <0					| Roundabout Mandatory							|
| <0	      			| Keep Left			    		 				|
| <0				    | Go Straight or Left  							|

For fourth and fifth images the model is very confident that its a Go Straight or left sign or Turn left ahead sign
| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 99.99         			| Go Straight or Left							| 
| <0     				| Turn Left Ahead								|
| <0					| Roundabout Mandatory							|
| <0	      			| Go Straight or Right			 				|
| <0				    | Slippery Road      							|

For the fifth image, the model is very confident its a Turn Left ahead sign and its correct.
| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 99.99         		| Turn Left Ahead								| 
| <0     				| Go Straight or Left							|
| <0					| Roundabout Mandatory							|
| <0	      			| Go Straight or Right  		 				|
| <0				    | Speed Limit 30km/h   							|

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


