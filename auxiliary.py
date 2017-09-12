import pickle
import numpy as np
from numpy.linalg import inv
import cv2
import glob
import matplotlib.pyplot as plt


''' Calculate calibration variables. '''

with open('distortion_var.p', 'rb') as handle:
    distortion_var = pickle.load(handle)

ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(distortion_var['objpoints'], distortion_var['imgpoints'], (1280, 720),None,None)

def undistort(img):
    global ret, mtx, dist, rvecs, tvecs
    return cv2.undistort(img, mtx, dist, None, mtx)


def threshold_image(image, channel, thresh = (0, 255), ft='RGB'):
    assert channel >=0 and channel <=2
    
    if ft == 'HLS':
        img = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    if ft == 'HSV':
        img = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    if ft == 'RGB':
        img = image
        
    s_channel = img[:,:,channel]
    binary = np.zeros_like(s_channel)
    binary[(s_channel > thresh[0]) & (s_channel <= thresh[1])] = 1
    return binary


''' Returns the directional gradient given an angle threshold '''

def dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi/2)):
    # Grayscale
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


''' Returns an image with the combine magnitude of gradients X & Y bounded by a threshold '''

def mag_threshold(img, sobel_kernel=3, thresh=(0, 255)):
    # Convert to grayscale
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
    binary_output[(gradmag >= thresh[0]) & (gradmag <= thresh[1])] = 1

    # Return the binary image
    return binary_output


''' Returns an image with the gradient accordingly to one direction bounded by a threshold '''

def single_axis_threshold(image, axis, sobel_kernel=3, thresh=(0,255)):
    assert axis == 'x' or axis == 'y'
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if axis == 'x':
        sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel) # Take the derivative in x
    elif axis == 'y':
        sobel = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel) # Take the derivative in y
        
    abs_sobel = np.absolute(sobel)    
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel)) 
    
    sbinary = np.zeros_like(scaled_sobel)
    sbinary[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1
    return sbinary


def binary_image(image):
    # Apply color filters
    image_HSV_V = threshold_image(image, channel=2, thresh=(200,255), ft='HSV')
    image_HLS_S = threshold_image(image, channel=2, thresh=(90,255), ft='HLS')
    # Combine color images
    combined_color = np.zeros_like(image_HSV_V)
    combined_color[((image_HSV_V == 1) & (image_HLS_S == 1))] = 1
    
    # Apply Gradient filters
    image_axis_x = single_axis_threshold(image, axis='x', sobel_kernel=5,thresh=(20,100))
    image_axis_y = single_axis_threshold(image, axis='y', sobel_kernel=3,thresh=(20,100))
    image_mag = mag_threshold(image, sobel_kernel=3,thresh=(30,100))
    image_dir = dir_threshold(image, sobel_kernel=3, thresh=(0.7, 1.3))
    # Combine Gradient images
    combined_grad = np.zeros_like(image_axis_x)
    combined_grad[((image_axis_x == 1) & (image_axis_y == 1)) 
              | ((image_mag == 1) & (image_dir == 1))] = 1

    # Combine color and gradient
    combined_all = np.zeros_like(combined_grad)
    combined_all[((combined_color == 1) | (combined_grad == 1))] = 1
    
    return combined_all



''' 
    This function takes an images and returns a bird-view perspective 
    of it with the transformation matrix used 
'''

def bird_eye(image):
#     top_left = (567,470)
#     top_right = (720,470)
#     bottom_left = (217,720)
#     bottom_right = (1110,720)
    top_left = (580,460)
    top_right = (707,460)
    bottom_left = (210,720)
    bottom_right = (1110,720)
    src = np.float32([[bottom_left, top_left, top_right, bottom_right]])

    left_bound, right_bound = (360, 970)
    top_left = (left_bound,0)
    top_right = (right_bound,0)
    bottom_left = (left_bound,720)
    bottom_right = (right_bound,720)
    dst = np.float32([[bottom_left, top_left, top_right, bottom_right]])

    img_x, img_y, chn = image.shape

    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(image, M, (img_y, img_x))
    return M,warped


''' 
The sliding window function breaks the lines in many spices in order to fit a polynomial
that will follow the line
'''

def sliding_window_polyfit(binary, nwindows=9, margin=100, minpix=50, plot=False):
    if plot:
        # Create an output image to draw on and  visualize the result
        out_img = np.dstack((binary, binary, binary))*255

    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    histogram = np.sum(binary[binary.shape[0]//2:,:], axis=0)
    midpoint = np.int(histogram.shape[0]/2)
    # Left Peak X position
    leftx_base = np.argmax(histogram[:midpoint])
    # Right Peak X position
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # Set height of windows
    window_height = np.int(binary.shape[0]/nwindows)

    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base

    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary.shape[0] - (window+1)*window_height
        win_y_high = binary.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        
        if plot:
            # Draw the windows on the visualization image
            cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),(0,255,0), 2) 
            cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),(0,255,0), 2) 
            
        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:        
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds] 

    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    
    # Generate x and y values for plotting
    ploty = np.linspace(0, binary.shape[0]-1, binary.shape[0] )
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    
    if plot:
        out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
        out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
        plt.imshow(out_img)
        plt.plot(left_fitx, ploty, color='yellow')
        plt.plot(right_fitx, ploty, color='yellow')
        plt.xlim(0, 1280)
        plt.ylim(720, 0)
    
    return left_fit, left_fitx, right_fit, right_fitx, ploty


'''
Once you already have a polynomial function is not necessary to use the sliding windows
anymore, instead a function that updates the polinommial with the new image should be enough
'''

def update_polyfit(binary, left_fit, right_fit):
    nonzero = binary.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    margin = 100
    left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] - margin)) & (nonzerox < (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] + margin))) 
    right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] - margin)) & (nonzerox < (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] + margin)))  

    # Again, extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]
    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    # Generate x and y values for plotting
    ploty = np.linspace(0, binary.shape[0]-1, binary.shape[0] )
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    
    return left_fit, left_fitx, right_fit, right_fitx, ploty


'''
Compute the vehicle curvature having the polynomial for each line
'''

def curvatures(y_eval, left_fitx, right_fitx, ploty):
    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/700 # meters per pixel in x dimension

    # Fit new polynomials to x,y in world space
    left_fit_cr = np.polyfit(ploty*ym_per_pix, left_fitx*xm_per_pix, 2)
    right_fit_cr = np.polyfit(ploty*ym_per_pix, right_fitx*xm_per_pix, 2)
    # Calculate the new radii of curvature
    left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
    
    # Now our radius of curvature is in meters
    return left_curverad, right_curverad


pixe2meters = {'y':30/720, 'x':3.7/700}


def curvature(y_eval, fitx, ploty):
    # Define conversions in x and y from pixels space to meters
    ym_per_pix = pixe2meters['y'] # meters per pixel in y dimension
    xm_per_pix = pixe2meters['x'] # meters per pixel in x dimension

    # Fit new polynomials to x,y in world space
    fit_cr = np.polyfit(ploty*ym_per_pix, fitx*xm_per_pix, 2)
    # Calculate the new radii of curvature
    curve_meters = ((1 + (2*fit_cr[0]*y_eval*ym_per_pix + fit_cr[1])**2)**1.5) / np.absolute(2*fit_cr[0])
    
    # Now our radius of curvature is in meters
    return curve_meters

''' Compute the distance from the lane to the center of the vehicle '''

vehicle_center = (640, 719)

def distance_center(fitx):
    return (fitx[vehicle_center[1]] - vehicle_center[0]) * pixe2meters['x']



def back_to_original(undist, warped, Minv, left_fitx, right_fitx, ploty):
    # Create an image to draw the lines on
    warp_zero = np.zeros_like(warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, (color_warp.shape[1], color_warp.shape[0])) 
    # Combine the result with the original image
    result = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)

    return result