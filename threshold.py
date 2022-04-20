import numpy as np
import cv2



def abs_sobel_thresh(img, orient='x', sobel_kernel=3, thresh=(0, 255)):

    # Apply threshold
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    #select the sobel mode 
    if orient == 'x':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel))
    if orient == 'y':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel))
    # Rescale back to 8 bit integer
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    binary_output = np.zeros_like(scaled_sobel)
    binary_output[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1

    # Return the result
    return binary_output# retuen  x or y


def mag_thresh(img, sobel_kernel=3, mag_thresh=(0, 255)):

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Take both Sobel x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Calculate the gradient magnitude
    gradmag = np.sqrt(sobelx**2 + sobely**2)
    # Rescale to 8 bit
    scale_factor = np.max(gradmag)/255 
    gradmag = (gradmag/scale_factor).astype(np.uint8) 
    # Create a binary image of ones where threshold is met, zeros otherwise
    binary_output = np.zeros_like(gradmag)
    binary_output[(gradmag >= mag_thresh[0]) & (gradmag <= mag_thresh[1])] = 1

    # Return the binary image
    return binary_output

def dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi/2)):

    # Apply threshold
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Calculate the x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Take the absolute value of the gradient direction, 
    # apply a threshold, and create a binary image result
    absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
    binary_output =  np.zeros_like(absgraddir)
    binary_output[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 1

    # Return the binary image
    return binary_output



def combined_thresholds(image, ksize = 3):
    # Choose a Sobel kernel size

    # Apply each of the thresholding functions
    gradx = abs_sobel_thresh(image, orient='x', sobel_kernel=ksize, thresh=(5, 100))
    grady = abs_sobel_thresh(image, orient='y', sobel_kernel=ksize, thresh=(5, 100))
    #c= cv2.bitwise_or(gradx,grady)
    
    mag_binary = mag_thresh(image, sobel_kernel=ksize, mag_thresh=(3, 255))
    dir_binary = dir_threshold(image, sobel_kernel=ksize, thresh=(0*np.pi/180, 90*np.pi/180)) 
    combined = np.zeros_like(dir_binary, np.uint8)    
    #combined[((gradx == 1) ) ] = 1
    combined[((gradx == 1) ) & ((mag_binary == 1) & (dir_binary == 1))] = 1
    #combined[ (mag_binary == 1) ] = 1

    return combined



def warp(image,state='in'):

    src = np.float32([[570,460],[image.shape[1] - 573,460],[image.shape[1] - 150,image.shape[0]],[150,image.shape[0]]])
    dst = np.float32([[200,0],[image.shape[1]-200,0],[image.shape[1]-200,image.shape[0]],[200,image.shape[0]]])
    if state == 'in':
        M = cv2.getPerspectiveTransform(src, dst)
    if state == 'out':
        M = cv2.getPerspectiveTransform(dst, src)
    
    warped = cv2.warpPerspective(image, M, (image.shape[1],image.shape[0]), flags=cv2.INTER_LINEAR)
    return warped




def filter(image):
    
    hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    s_channel = hls[:,:,2]
    l_channel = hls[:,:,1]

    sxbinary = combined_thresholds(image)


    # Threshold color channel
    s_thresh_min = 80
    s_thresh_max = 255
    l_thresh_min = 190
    l_thresh_max = 255  
    
    
    s_binary = np.zeros_like(s_channel)
    s_binary[((s_channel >= s_thresh_min) & (s_channel <= s_thresh_max)) ] = 1
    
    l_binary = np.zeros_like(s_channel)
    l_binary[((l_channel >= l_thresh_min) & (l_channel <= l_thresh_max))] = 1

    # Combine the two binary thresholds
    combined_binary = np.zeros_like(sxbinary)
    #combined_binary [(l_binary == 1)& (sxbinary == 1)] = 1
    combined_binary[((s_binary == 1) & (sxbinary == 1)) | ((sxbinary == 1) & (l_binary == 1))] = 1
    
    return combined_binary