

## project pipeline 

1- forward perspective transform:

we can transform road image to a bird-eye view image, in which, it is easier to detect the curving angles of lanes. 
The code for Perspective Transformation is contain in the threshold.py . 

2- thresholding:
 
 we use a cascade filter consisting of two functions the first is Gradient thresholding and the second is HLS thresholding.we easily can change the saturation and lightness factors according to the environment 
The code for Perspective Transformation is contain in the threshold.py . 

4- sliding window search: 

when we detected the lane lines in the frame, we can use the last ten frames information and use a sliding window, placed around the line centers, to find and follow lane lines from bottom to the top of the image/frame.
The code for Perspective Transformation is contain in the saving.py . 

 
5- illustrating lane lines on image




