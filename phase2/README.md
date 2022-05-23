
# Vehicle Detection



## target
locate and identify the cars on the road using HOG feature to search accross the image 


## pipeline 

The project go through the followig steps:


Step 1: Create a function to compute Histogram of Oriented Gradients on image dataset.

Step 2: Extract HOG features from training images and build car and non-car datasets to train a classifier.

Step 3: Train a classifier to identify images of vehicles.

step 4:loop in the frame with the sliding window and get Hog feature for each window 

Step 5: Create a function to draw bounding rectangles based on detection of vehicles.

Step 6: Identify vehicles within images of highway driving.

Step 7: Track images across frames in a video stream
## pipeline 

The project go through the followig steps:


Step 1: Create a function to compute Histogram of Oriented Gradients on image dataset.

Step 2: Extract HOG features from training images and build car and non-car datasets to train a classifier.

Step 3: Train a classifier to identify images of vehicles.

step 4:loop in the frame with the sliding window and get Hog feature for each window 

Step 5: Create a function to draw bounding rectangles based on detection of vehicles.

Step 6: Identify vehicles within images of highway driving.

Step 7: Track images across frames in a video stream
## Installation


1- download the notebook 

2- put the notebook with the same directory with the dataset 

3- for traing the model run the cell called train the model with MLP
or 
easily you can load the trained mode by running this cell 

```bash
mlp = joblib.load('mlp1.pkl')
X_scaler = joblib.load('scaler1.pkl')
```
 4- finally put the input video directory inside this line 

```bash
 clip1 = VideoFileClip('put the video path here ')
 
```
    