##Writeup Template
###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./pipeline_images/hog_vis.png
[image2]: ./pipeline_images/scale1.png
[image3]: ./pipeline_images/scale15.png
[image4]: ./pipeline_images/scale2.png
[image5]: ./pipeline_images/scale35.png
[image6]: ./pipeline_images/pipeline1.png
[image7]: ./pipeline_images/pipeline2.png
[image8]: ./pipeline_images/pipeline3.png
[image9]: ./pipeline_images/pipeline4.png
[image10]: ./pipeline_images/pipeline5.png
[image11]: ./pipeline_images/pipeline6.png
[image12]: ./pipeline_images/test_images_result.png
[video1]: ./output_video/out_project_video.mp4
[video2]: ./output_video/out_project_video_findlane.mp4

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  
You can submit your writeup as markdown or pdf.  
[Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

###Histogram of Oriented Gradients (HOG)

The code is written as continuation from Project 4 (Advanced Lane Finding)
Folder findlane contains the P4 code as module
Folder detection contains P5 code as module as well
cli.py is the script that uses the module to produce the combined result
`python ./cli.py --cal 1` calibrates the camera and undistorts test images
`python ./cli.py --imgd 1` executes the detection pipeline on test images
`python ./cli.py --vidd 1` executes findlane (optionally) and detection on project video

####1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the class Features in `detection/features.py`).  

I started by reading in all the `vehicle` and `non-vehicle` images.  
Here is an example of one of each of the `vehicle` and `non-vehicle` classes, along with their HOG representation:

![alt text][image1]

I then explored different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  
I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

The previous image is using the `YUV` color space and HOG parameters of `orientations=11`, `pixels_per_cell=(16, 16)` and `cells_per_block=(2, 2)`
I also used `hog_channels='ALL'`. This fact helped a lot achieving high accuracy for training the SVM later, at the cost of speed. 
Increasing `pixels_per_cell` from 8 to 16 led to better training time without noticeable difference in accuracy.

####2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters and chose the values above as the best trade-off between high accuracy and execution time

####3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM using only HOG features, in class Classifier, file `detection/classifier.py` line 98. 
In fact, i wrote code to extract HOG features once, saved them using pickle, and used the pickled file to load features each time i needed to 
experiment with the pipeline (lines 49-94 in same file). Even though that the accuracy is different between runs, it is maintained over 98% in most cases, even reaching 98.4%.
 

###Sliding Window Search

####1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I decided to search window positions at different scales all over the image and came up with using the lower half of the image as described in lesson.
I selected the scale through experimentation, but the general idea was to select scales for specific parts of the image based on the possibility to exist
a car in the specific scale. For example low scale (1.0) was used in the upper part of the area under consideration as shown in the next image. I also decided to 
overlap as little as possible, 50% horizontally and 75% vertically proved to be a good choice. Mostly, i used the function find_cars as described in the lesson
with little modifications, supposing that i use 'ALL' three hog channels

![alt text][image2]

Scale 1.5
![alt text][image3]

Scale 2.0
![alt text][image4]

Scale 3.5
![alt text][image5]

####2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on four scales using YUV 3-channel HOG features, which provided a nice result.  

Here are some example images:

![alt text][image12]
---

### Video Implementation

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](https://github.com/antalakas/carnd-term1-p5/blob/master/output_video/out_project_video.mp4)
Please also note that i integrated findlane andvehicle detection [in this video](https://github.com/antalakas/carnd-term1-p5/blob/master/output_video/out_project_video_findlane.mp4)


####2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

The pipeline works as follows:

The original image
![alt text][image6]

find_cars outputs predictions (I recorded the positions of positive detections in each frame of the video.)
![alt text][image7]

A heatmap is calculated for the image, based on the fact that the predictions included multiple rectangles in overlapping positions
![alt text][image8]

The heatmap is thresholded, in the specific example threshold=2, which means that we persist output if at least 2 overlapping rectangles exist over a specific area
![alt text][image9]

I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.
![alt text][image10]

Final rectangles are drawn over the original image having their center calculated over the labels (class Painter, in painter.py, function draw_labeled_bboxes)
![alt text][image11]

---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

I found in references that the HOG method is around since 2005, also the lesson somehow leads to choose HOG as the method of implementation for the project. Probably, integrating 
more features based on histogram and color in the feature vector could improve the accuracy. It seems to me that current implementation is not fast enough and i could do better limiting 
the number of sliding windows. Also, simply producing a video based on the pipeline described was not smooth, in `detection.py, line 58, function add_bbox_history` the set window rectangles
is added to frame history (value of 10 was selected for the previous frames used) and the final heatmap calculated for the image is taking all these boxes into account. I had problems 
deciding about the threshold to use, i used constant values like 10 or 5 depending on the history depth, in the end i used variable threshold `detection.py, line 169` 
(as described here https://github.com/jeremy-shannon/CarND-Vehicle-Detection/blob/master/vehicle_detection_project.ipynb). 
In the video the white car detection is lost for sub-second time, this means that the scale and overlap of the sliding windows 
can be improved. Scales lower than 1.0 for the sliding window output more false positives, probably further optimizing the classifier as well as using more training images could improve 
the detection further in the line of sight. I also had a look in the YOLO implementation, probably this is the way to go, but still it is well worth understanding the classic methods.