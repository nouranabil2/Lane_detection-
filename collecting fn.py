import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import cv2
from IPython.core.display import Video
import glob
from moviepy.editor import *
first_half1=VideoFileClip('c3.mp4')
first_half2=VideoFileClip('c4.mp4')
second_half1=VideoFileClip('c2.mp4')
second_half3=VideoFileClip('c1.mp4')
first_half=clips_array([[first_half1,first_half2]])
second_half=clips_array([[second_half1,second_half3]])
video=clips_array([[first_half.set_position(600,400)],[second_half.set_position(600,400)]])
video.ipython_display(width=1200,height=700)
video.write_videofile('Total.mp4')