#from unittest import result
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import cv2
from IPython.core.display import Video
import glob
from moviepy.editor import *
from threshold import *
from lane_drawing import*

lane_detector = LaneDetector()
draw_lane = lane_detector.draw_lane
#function returning final image
def process_image(image):
    #applying filter to get the edges
    filtered_binary = filter(image)
    #applying wrapfunction to get the prespective
    binary_warped = warp(filtered_binary,state='in')
    #convert to grey scale to be able to draw
    filtered_binary_grayscale=filtered_binary*255
    binary_warped_grayscale=binary_warped*255
    #applying draw fn to get the lane
    final_image,polyimg = draw_lane(image, binary_warped, filtered_binary)
    return final_image

#function returning polynomyal image
def poly(image):
   filtered_binary = filter(image)
   binary_warped = warp(filtered_binary)
   final_image,polyimg = draw_lane(image, binary_warped, filtered_binary)
   return polyimg

#combining all processes of the video in one video
def combine():

    first_half1=VideoFileClip('out1.mp4')
    first_half2=VideoFileClip('out2.mp4')
    second_half1=VideoFileClip('out4.mp4')
    second_half3=VideoFileClip('out3.mp4')
    first_half=clips_array([[first_half1,first_half2]])
    second_half=clips_array([[second_half1,second_half3]])
    video=clips_array([[first_half.set_position(600,400)],[second_half.set_position(600,400)]])
    video.ipython_display(width=1200,height=700)
    video.write_videofile(strout)
    return print('Done')

#converting the filtered image from binary to RGB 
def covert_filter(image):
  filtered_binary = filter(image)
  filtered_binary_grayscale=filtered_binary*255
  gtb=cv2.cvtColor(filtered_binary_grayscale,cv2.COLOR_GRAY2RGB)
  return gtb    

#converting the prespective image from binary to RGB 
def covert_warp(image):
  filtered_binary = filter(image)
  binary_warped = warp(filtered_binary)
  warp_grayscale=binary_warped*255
  gtb1=cv2.cvtColor(warp_grayscale,cv2.COLOR_GRAY2RGB)
  return gtb1

#producing final video accourding to the mode input
def saving(strin,strout,mode):
    
    clip1 = VideoFileClip(strin)
    output = strout
    if mode==0:
        #saving final output video 
        finalresult = clip1.fl_image(process_image) 
        finalresult.write_videofile(output, audio=False)
    if mode==1:
        #saving final output video 
        finalresult = clip1.fl_image(process_image) 
        finalresult.write_videofile('out1.mp4', audio=False)
        #saving filtered output video 
        filteringresult = clip1.fl_image(covert_filter) 
        filteringresult.write_videofile('out2.mp4', audio=False)
        #saving prespective output video 
        prespectiveresult = clip1.fl_image(covert_warp) 
        prespectiveresult.write_videofile('out3.mp4', audio=False)
        #saving polyshape output video 
        polyshape = clip1.fl_image(poly) 
        polyshape.write_videofile('out4.mp4', audio=False)
        #combining the 4 previous videos together
        combined_result=combine()


def main():
    args = sys.argv
    #input Path 
    strin= args[1]
    #output path
    strout=args[2]
    #mode number
    mode = int(args[4])
    saving(strin,strout,mode)

if __name__ == "__main__":
    main()
