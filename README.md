[Project Video](https://vimeo.com/232379970) [Debug](https://vimeo.com/232392603)

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with re spect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./output_images/undistort_output.png "Undistorted"
[image2]: ./output_images/road_undistort.png "Road Transformed"
[image3]: ./output_images/binary.png "Binary Example"
[image4]: ./output_images/warp_template.png "Transformation Template"
[image5]: ./output_images/warped.png "Warped"
[image6]: ./output_images/warped_binary.png "Binary Warped"
[image7]: ./output_images/histogram.png "Histogram"
[image8]: ./output_images/sliding_window.png "Sliding Window"
[image9]: ./output_images/marked_image.png "Road Detection"
[video1]: ./Lane_Detection.mp4 "Video"


## Contents
* [Line_Detection_Pipeline](Line_Detection_Pipeline): This notebook contains my step by step experimentations in achieving the project.
* [Create_Video_Output](Create_Video_Output.ipynb): This notebook creates the video for the solution of this project.
* [auxiliary.py](auxiliary.py): This python file contains the functions for creating the solution video.
* [Project Video](https://vimeo.com/232379970)


### 1. Camera Calibration
The code for this step is contained in the first code cell of the IPython notebook located in "./examples/example.ipynb" (or in lines # through # of the file called `some_file.py`).  

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I created the function `undistort(image)` to be used through the project, which applies the distortion correction to the image using the `cv2.undistort()`. See the example bellow: 

![alt text][image1]


### Pipeline (single images)

#### 1. Road Image Undistorted

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
![alt text][image2]

#### 2. Creating a Binary Image

I used a combination of color and gradient thresholds to generate a binary image (for details on my experimentation see [Pipeline](./Line_Detection_Pipeline.ipynb)).  The function bellow receives a RBG image and converts to the appropriate encoding (RGB, HLS, HSV) and filter the pixels of the chosen channel according to a threshold.

```python
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
```

The three next functions apply the same idea on different ways to retrieve the image gradient (axis, magnitude, direction).

```python
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
```
Finally, after combining all gradients and color schemes are combined:

```python
def binary_image(image):
    # Apply color filters
    image_HSV_V = threshold_image(image, channel=2, thresh=(220,255), ft='HSV')
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
```
Here's an example of my output for this step.

![alt text][image3]

#### 3. Perspective Warping (Bird-Eye)
The code bellow performs a perspective transform including a function called `cv2.warpPerspective()`.  The function calculate the transformation matrix between 4 points in the original image as a trapezoid and how the will be if we were to see the picture from the sky in a straight line (hence bird-eye). The image transformation template helps to understand the idea.

```python
def bird_eye(image):
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
```

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 580, 460      | 360, 0        | 
| 707, 460      | 970, 0        |
| 210, 720      | 360, 720      |
| 1110, 720     | 970, 720      |

![alt text][image4]

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image5]


#### 4. Identifying Lines

Let's apply the binary view on our warped image: 
![alt text][image6]

In order to find the lanes I applied two concepts. First I created a histogram where **x-axis** tells the column horizontally in my binary picture and the **y-axis** tells me how many white pixels are in that column.  Thus, for the binary picture above we obtain the histogram bellow.

![alt text][image7]

The second concept is called "sliding windows". The idea is to slice the picture in windows of the same size and mark the windows that achieve a certain number of  white pixels using the histogram. Then, the points of the windows selected can be used in order to fit a polynomial of the second degree that will be used to draw the lane.

![alt text][image8]

Now, with the line identified we just have to warp back to our original perspective to achieve a desireble output.

![alt text][image9]

#### 5. Radius of Curvature and Position of the vehicle with Respect to Center.
In order to be able to debug and know if good lane was detected it was necessary to calculate the curvature and the distance form the lane to the center of the car.
This was achivable with the folloing functions:
```python
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
```

## Discussion

The biggest challenge in detecting lines happens when shades or excessive brightness appear on the road. Moving foward a way to make the pipeline more robust to these changes could be find a good combination of the channels in LUV and LAB color spaces, since both of these schemes have a dedicated channel for ilumination.

