import numpy as np
import cv2
from threshold import *

class LaneDetector:

    def __init__(self):
        self.detected = True
        self.left_lane_inds = []  # Create empty lists to receive left and right lane pixel indices
        self.right_lane_inds = []

        self.n_frames = 10

        # x values of the last n fits of the line
        self.recent_xfitted = []
        # average x values of the fitted line over the last n iterations
        self.bestx = [np.zeros_like(720, np.float32), np.zeros_like(720, np.float32)]

        # coefficient values of the last n fits of the line
        self.recent_coefficients = []

        # polynomial coefficients averaged over the last n iterations
        self.best_fit = [0, 0, 0]

        self.vehicle_offset = 0.0
        self.avg_curverad = 1000

    def draw_lane(self, orignal_image, binary_warped, filtered_binary):

        nonzero = binary_warped.nonzero()  # Identify the x and y positions of all nonzero pixels in the image
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        # Create an output image to draw on and  visualize the result
        # out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255

        margin = 50
        # Set minimum number of pixels found to recenter window
        minpix = 50

        if self.detected:
            # Take a histogram of the bottom half of the image
            histogram = np.sum(binary_warped[binary_warped.shape[0] // 2:, :], axis=0)

            # Find the peak of the left and right halves of the histogram
            # These will be the starting point for the left and right lines
            midpoint = np.int(histogram.shape[0] / 2)
            leftx_base = np.argmax(histogram[:midpoint])
            rightx_base = np.argmax(histogram[midpoint:]) + midpoint

            nwindows = 9  # Choose the number of sliding windows
            window_height = np.int((binary_warped.shape[0]) / nwindows)  # Set height of windows

            leftx_current = leftx_base  # Current positions to be updated for each window
            rightx_current = rightx_base  # Set the width of the windows +/- margin

            left_lane_inds = []
            right_lane_inds = []

            # Step through the windows one by one
            for window in range(nwindows):

                # Identify window boundaries in x and y (and right and left)
                win_y_low = binary_warped.shape[0] - (window + 1) * window_height
                win_y_high = binary_warped.shape[0] - window * window_height
                win_xleft_low = leftx_current - margin
                win_xleft_high = leftx_current + margin
                win_xright_low = rightx_current - margin
                win_xright_high = rightx_current + margin

                # Identify the nonzero pixels in x and y within the window
                good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                                  (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
                good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                                   (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]

                # Append these indices to the lists
                left_lane_inds.append(good_left_inds)
                right_lane_inds.append(good_right_inds)
                # If you found > minpix pixels, recenter next window on their mean position
                if len(good_left_inds) > minpix:
                    leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
                if len(good_right_inds) > minpix:
                    rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

            # Concatenate the arrays of indices
            self.left_lane_inds = np.concatenate(left_lane_inds)
            self.right_lane_inds = np.concatenate(right_lane_inds)
        else:
            self.left_lane_inds = ((nonzerox > (self.best_fit[0][0] * (nonzeroy ** 2) + self.best_fit[0][1] * nonzeroy +
                                                self.best_fit[0][2] - margin)) & (
                                               nonzerox < (self.best_fit[0][0] * (nonzeroy ** 2) +
                                                           self.best_fit[0][1] * nonzeroy + self.best_fit[0][
                                                               2] + margin)))
            self.right_lane_inds = (
                        (nonzerox > (self.best_fit[1][0] * (nonzeroy ** 2) + self.best_fit[1][1] * nonzeroy +
                                     self.best_fit[1][2] - margin)) & (
                                    nonzerox < (self.best_fit[1][0] * (nonzeroy ** 2) +
                                                self.best_fit[1][1] * nonzeroy + self.best_fit[1][2] + margin)))

        # Again, extract left and right line pixel positions
        leftx = nonzerox[self.left_lane_inds]
        lefty = nonzeroy[self.left_lane_inds]
        rightx = nonzerox[self.right_lane_inds]
        righty = nonzeroy[self.right_lane_inds]

        ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])

        # Fit a second order polynomial to each
        if lefty.shape[0] >= 400 and righty.shape[0] >= 400 and leftx.shape[0] >= 400 and rightx.shape[0] >= 400:
            self.detected = False
            self.left_fit = np.polyfit(lefty, leftx, 2)
            self.right_fit = np.polyfit(righty, rightx, 2)

            if len(self.recent_coefficients) >= self.n_frames:
                self.recent_coefficients.pop(0)
            self.recent_coefficients.append([self.left_fit, self.right_fit])

            self.best_fit = [0, 0, 0]
            for coefficient in self.recent_coefficients:
                self.best_fit[0] = self.best_fit[0] + coefficient[0]
                self.best_fit[1] = self.best_fit[1] + coefficient[1]

            self.best_fit[0] = self.best_fit[0] / len(self.recent_coefficients)
            self.best_fit[1] = self.best_fit[1] / len(self.recent_coefficients)

            # Generate x and y values for plotting
            left_fitx = self.best_fit[0][0] * ploty ** 2 + self.best_fit[0][1] * ploty + self.best_fit[0][2]
            right_fitx = self.best_fit[1][0] * ploty ** 2 + self.best_fit[1][1] * ploty + self.best_fit[1][2]

            if len(self.recent_xfitted) >= self.n_frames:
                self.recent_xfitted.pop(0)

            self.recent_xfitted.append([left_fitx, right_fitx])

            self.bestx = [np.zeros_like(720, np.float32), np.zeros_like(720, np.float32)]
            for fit in self.recent_xfitted:
                self.bestx[0] = self.bestx[0] + fit[0]
                self.bestx[1] = self.bestx[1] + fit[1]

            self.bestx[0] = self.bestx[0] / len(self.recent_xfitted)
            self.bestx[1] = self.bestx[1] / len(self.recent_xfitted)


        else:
            self.detected = True

        # Create an image to draw on and an image to show the selection window
        out_img = np.dstack((binary_warped, binary_warped, binary_warped)) * 255

        window_img = np.zeros_like(out_img)

        # Generate a polygon to illustrate the search window area
        # And recast the x and y points into usable format for cv2.fillPoly()
        margin = 5
        left_line_window1 = np.array([np.transpose(np.vstack([self.bestx[0] - margin, ploty]))])
        left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([self.bestx[0] + margin,
                                                                        ploty])))])
        left_line_pts = np.hstack((left_line_window1, left_line_window2))

        right_line_window1 = np.array([np.transpose(np.vstack([self.bestx[1] - margin, ploty]))])
        right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([self.bestx[1] + margin,
                                                                         ploty])))])
        right_line_pts = np.hstack((right_line_window1, right_line_window2))

        center_line_window1 = np.array([np.transpose(np.vstack([self.bestx[0] + margin, ploty]))])
        center_line_window2 = np.array([np.flipud(np.transpose(np.vstack([self.bestx[1] - margin, ploty])))])
        center_line_pts = np.hstack((center_line_window1, center_line_window2))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(window_img, np.int_([left_line_pts]), (255, 255, 0))
        cv2.fillPoly(window_img, np.int_([right_line_pts]), (0, 0, 255))
        cv2.fillPoly(window_img, np.int_([center_line_pts]), (0, 255, 0))

        window_img_unwrapped = warp(window_img, state='out')

        result = cv2.addWeighted(orignal_image, 1, window_img_unwrapped, 0.3, 0)

        return result, window_img
