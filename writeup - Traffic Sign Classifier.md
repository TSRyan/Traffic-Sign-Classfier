#**Traffic Sign Recognition** 

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # "Image References"

[image1]: data_summary.png "Visualization"
[image2]: class_examples.png "Class examples"
[image3]: class_examples_gray.png "Grayscaled examples"
[image4]: class_examples_equalized.png "Equalization"
[image5]: ./web_images/general_caution.jpg "Traffic Sign 1"
[image6]: ./web_images/no_entry.jpg "Traffic Sign 2"
[image7]: ./web_images/road_work.jpg "Traffic Sign 3"
[image8]: ./web_images/stop.jpg "Traffic Sign 4"
[image9]: ./web_images/yield.jpg "Traffic Sign 5"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/TSRyan/Traffic-Sign-Classfier/blob/master/Traffic_Sign_Classifier.ipynb).

###Data Set Summary & Exploration

####1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799 samples
* The size of the validation set is 4410 samples
* The size of test set is 12630 samples
* The shape of a traffic sign image is 32x32x3
* The number of unique classes/labels in the data set is 43

####2. Include an exploratory visualization of the dataset.

Below is an exploratory visualization of the data set. It is a bar chart showing how the data in each subset compare to the overall distribution.  The frequency of each class is given as is the $\chi^2$ goodness-of-fit test statistic and statistical significance (p-value).  Note that this is the proper statistical test to perform on unordered categorical data. The $\chi^2$ and p values  indicate that the distributions are not statistically different from one another, which is desirable as the training, validation, and test sets should all be representative of the population.  In this case, I do not know the proportion of each class in the population, so the analysis is based on the total data set.  Ideally, the spread of frequencies would not be so drastic as in this case of this data set.  Classes that appear under-represented could be augmented with randomized transformations of the image samples, but that was unnecessary to achieve the desired accuracy level for this project.

![alt text][image1]

###Design and Test a Model Architecture

####1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to convert the images to grayscale because ...

Here is an example of a few traffic sign images before and after grayscaling.

![alt text][image2]

![alt text][image3]

Then the image histograms were equalized, using contrast-limited adaptive histogram equalization (CLAHE).  The goal was to handle some of the difference in lighting conditions between the image samples.  The result of this step is shown with the examples from above:

![Equalized images][image4]

Although it doesn't totally mitigate the differences in lighting conditions, the CLAHE certainly makes some of the samples clearer as evidenced by the 70 kph road sign above. 

As previously mentioned data augmentation would further improve the training set by increasing the number of samples in the underrepresented classes, but it was an unnecessary complication in reaching the project goal.

####2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

|      Layer      |               Description                |
| :-------------: | :--------------------------------------: |
|      Input      |            32x32x1 gray image            |
| Convolution 9x9 | 1x1 stride, valid padding, outputs 24x24x20 |
|      RELU       |                                          |
|     Dropout     |           0.7 keep probability           |
| Max pooling 2x2 | 2x2 stride, valid padding, outputs 12x12x20 |
| Convolution 5x5 | 2x2 stride,  valid padding, outputs 8x8x16 |
|      RELU       |                                          |
|     Dropout     |           0.7 keep probability           |
| Max pooling 2x2 | 2x2 stride, valid padding, outputs 4x4x16 |
|     Flatten     |               outputs 256                |
| Fully connected |                output 160                |
|      RELU       |                                          |
| Fully connected |                output 84                 |
|      RELU       |                                          |
|     Output      |      outputs 43 (number of classes)      |

####3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used backpropagation with the Adam optimizer.  I allowed 20 epochs to ensure the solution was done converging.  .  The batch size was 128.  The results were not sensitive to the batch size, so I left it as the default value. Through trial-and-error, I decided on a learning rate of 0.001 (the default value) and a keep probability of 0.7 for the dropout layers.

####4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 99.9%
* validation set accuracy of 96.0%
* test set accuracy of 95.3%

*If a well known architecture was chosen:*
* *What architecture was chosen?*
* *Why did you believe it would be relevant to the traffic sign application?*
* *How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?*


I used the LeNet architecture as a starting point.  Because the architecture has already been proven to work for text recognition, I thought it would be appropriate for reading speed limit signs.  I immediately added two dropout layers to prevent overfitting of the model to the training data because the training accuracy was acceptable but the  model did not perform well enough on the validation set.  The initial layer sizes did not allow convergence to an accurate model, so I increased the size of the convolutions as well as the depths.  Furthermore, the other layers required adjustment to fit the new convolutions as well as the change in output classes.  I chose the sizes of the layers to successively compress the data space to the output size without dropping too much information at any given transition between layers. 

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![General Caution][image5] ![No Entry][image6] ![Road Work][image7] ![Stop][image8] ![Yield][image9]

The images are relatively tightly cropped to the signs, which makes classification easier, but the backgrounds of some of them can pose an issue.  Particularly, the road work sign has many dark and light patches in the background that could confuse a classifier.  The many lines of the trees in the background of the stop sign could also be distracting to CNN classification, but overall the images should not be too difficult for a well-trained classifier.

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

|      Image      |   Prediction    |
| :-------------: | :-------------: |
| General caution | General caution |
|    No entry     |    No entry     |
|    Road work    |    Road work    |
|      Stop       |      Stop       |
|      Yield      |      Yield      |


The model was able to correctly guess 5 of the 5 traffic signs, which gives an accuracy of 100%. This compares comparably to the accuracy on the test set of ~95%.  To better determine the real-world function of the final model, a much larger set would be required.  The test set gives an estimate of the real-world accuracy, but a completely novel test would provide a better estimate as it would be unseen, even by the developer, before implementation.  As there were only 5 new images, an accuracy of 100%  or 80% would be reasonable with respect to the web samples.

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 10th cell of the Ipython notebook.

For the first image, the model is pretty sure that this is a general caution sign (probability of 0.99), and the image does contain a general caution sign. The top five soft max probabilities were:

| Probability |              Prediction               |
| :---------: | :-----------------------------------: |
|    0.994    |            General caution            |
|   6.44e-3   |            Traffic signals            |
|   3.15e-6   |               Road work               |
|   1.27e-6   |             Priority road             |
|   1.22e-6   | Right-of-way at the next intersection |

For the second image the softmax probabilities were:

| Probability |      Prediction      |
| :---------: | :------------------: |
|    0.999    |       No entry       |
|   2.38e-4   |      No passing      |
|   8.46e-5   |         Stop         |
|   2.18e-5   | Go straight or right |
|   1.36e-6   |   Turn left ahead    |

for the third image:

| Probability |          Prediction          |
| :---------: | :--------------------------: |
|    0.992    |          Road Work           |
|   7.12e-3   |        Slippery Road         |
|   3.10e-4   |          Keep right          |
|   3.02e-4   | Dangerous curve to the right |
|   2.87e-4   |      Beware of ice/snow      |

for the fourth image:

| Probability |      Prediction      |
| :---------: | :------------------: |
|    0.994    |         Stop         |
|   4.33e-3   |      Keep right      |
|   7.43e-4   | Speed limit (80km/h) |
|   3.48e-4   |        Yield         |
|   1.42e-4   | Go straight or right |

and for the fifth image:

| Probability |      Prediction      |
| :---------: | :------------------: |
|    0.999    |        Yield         |
|   5.85e-7   |      Keep left       |
|   1.04e-8   | Speed limit (80km/h) |
|   9.70e-9   | Speed limit (60km/h) |
|  4.16e-10   |      Road work       |

In all of these cases, the predictive model is remarkably certain of the predicted classes.

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
####1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?

