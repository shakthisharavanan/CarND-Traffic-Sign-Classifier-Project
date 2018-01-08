# **Traffic Sign Recognition** 

Here is a link to my [project code](https://github.com/shakthisharavanan/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)


## Writeup




[//]: # (Image References)

[image1]: ./histogram.png "Histogram"
[image2]: ./random.png "Random train image"
[image3]: ./gray.png "Grayscale"
[image4]: ./1.jpg "Turn Left Ahead"
[image5]: ./2.jpg "Right of way at the next intersection"
[image6]: ./3.jpg "No entry"
[image7]: ./4.jpg "Yield"
[image8]: ./5.png "go straight or right"


---


### Data Set Summary & Exploration


I used numpy and matplotlib libraries to explore the traffic signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12360
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2. Visualization of the dataset.

Here is an exploratory visualization of the data set. It is a histogram plot of the training data ...

![alt text][image1]

Here is a random image from the training data ...

![alt text][image2]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

At first, I shuffled the dataset for a better network performance. Then I decided to convert the images to grayscale because its simpler and faster for further operations. It also reduces the size of the data.

Here is an example of a traffic sign image before and after grayscaling.

![alt text][image2]

![alt text][image3]

As a last step, I normalized the image so that the data has mean zero and equal variance so reduced skewness.


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model(after preprocessing) consisted of the following layers:

| Layer         		|     Description	        					                | 
|:---------------------:|:---------------------------------------------:                | 
| Input         		| 32x32x1 Grayscale image   					                | 
| Convolution 5x5     	| 1x1 stride, VALID padding, outputs 28x28x6 	                |
| RELU					| outputs 28x28x6								                |
| Max pooling	      	| 2x2 stride, 2x2 kernel, Valid padding, outputs 14x14x6 		|
| Convolution 5x5	    | 1x1 stride, VALID padding, outputs 10x10x16  	                |
| RELU					| outputs 10x10x16								                |
| Max pooling	      	| 2x2 stride, 2x2 kernel, Valid padding, outputs 5x5x16 		|
| Flatten   	      	| outputs 400       	                                        |
| Fully connected		| input 400, outputs 120        								|
| RELU  				| outputs 120               									|
| Fully connceted		| input 120, outputs 84											|
| RELU					| outputs 84     												|
| Fully connected		| input 84, outputs 43											|


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used a GeForce GTX 1060 GPU.

Learning rate : 0.0015

Batch size : 128

Epochs : 60

Optimizer : Adam

I had to play around with the learning rate(started with 0.005) and epochs(started with 10) before setling at the above mentioned values to get the validation accuracy over 93%.


#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

I used the well-known, proven-to-work LeNet architecture. Implementation of Lenet was straightforward and it trains relatively faster. It worked well with the MNIST dataset, so I went ahead with this architecture. I believed that it would extract interesting features from the traffic sign images as it did with the MNIST dataset.

My final model results were:
* training set accuracy of 0.998
* validation set accuracy of 0.943
* test set accuracy of 0.922

 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

The first and the fourth image might be difficult to classify because beacuse of the irrelevant information like the watermark in the images.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Turn Left ahead     		| Right-of-way at the next intersection  | 
| Right-of-way at the next intersection    			| Right-of-way at the next intersect	|
| No entry				| No entry											|
| Yield      		| Yield					 				|
| Go straight or right			| Go straight or right      							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. Given that these are not clean images, this compares favorably to the accuracy on the test set of 92.2%.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)


For the first image, the model wrongly predicts a turn Left ahead sign as a Right-of-way at the next intersection sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0.99757         			| Right-of-way at the next intersection									| 
| 0.00243     				| Traffic signals										|
| 0.00000					| General caution										|
| 0.00000	      			| Turn left ahead					 				|
| 0.00000				    | Beware of ice/snow      							|


For the second image, the model is sure that this is a Right-of-way at the next intersection sign (probability of 1), and the image does contain a Right-of-way at the next intersection sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.00000         			| Right-of-way at the next intersection									| 
| 0.00000     				| Priority road										|
| 0.00000					| Double curve										|
| 0.00000	      			| Roundabout mandatory					 				|
| 0.00000				    | Speed limit (20km/h)     							|

For the third image, the model is sure that this is a No entry sign (probability of 1), and the image does contain a No entry sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.00000         			| No entry								| 
| 0.00000     				| Stop										|
| 0.00000					| Traffic signals										|
| 0.00000	      			| Speed limit (20km/h)					 				|
| 0.00000				    | Speed limit (30km/h)      							|

For the fourth image, the model is sure that this is a Yield sign (probability of 1), and the image does contain a Yield sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.00000         			| Yield								| 
| 0.00000     				| Stop										|
| 0.00000					| Priority road										|
| 0.00000	      			| Keep left					 				|
| 0.00000				    | No passing     							|

For the fifth image, the model is sure that this is a Go straight or right sign (probability of 1), and the image does contain a Go straight or right sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.00000         			| Go straight or right									| 
| 0.00000     				| Turn left ahead										|
| 0.00000					| General caution										|
| 0.00000	      			| Road work					 				|
| 0.00000				    | no entry     							|

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


