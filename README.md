
## Project Pipeline 

1- Forward perspective transform:

We can transform road image to a bird-eye view image, in which, it is easier to detect the curving angles of lanes. 
The code for Perspective Transformation is contain in the threshold.py . 

2- Thresholding:
 
 We use a cascade filter consisting of two functions the first is Gradient thresholding and the second is HLS thresholding.we easily can change the saturation and lightness factors according to the environment 
The code for Perspective Transformation is contain in the threshold.py . 

4- Sliding window search: 

When we detected the lane lines in the frame, we can use the last ten frames information and use a sliding window, placed around the line centers, to find and follow lane lines from bottom to the top of the image/frame.
The code for Perspective Transformation is in the lane_drawing.py . 

 
5- Illustrating lane lines on image

## How to run
1) If you're using Windows install WSL
2) Open the project folder and open bash propmt by writing "bash" instead of the path on the top

3) Write the following command line








```bash
 bash run.sh [path of input video] [path of output] --dubug [mode]
 
```
Note:

If you want the final video set the mode to 0

If you want the debug video set the mode to 1

Example
```bash
  bash run.sh challenge_video.mp4 out.mp4 --dubug 0
```


Since test videos are in the same folder you can write the name of the file right away but write the full path if it's outside the folder
    