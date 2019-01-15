import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from line import Line
import glob
import os

# image tuning variables
sobel_kernel = 7
sobelX_thresh = (75, 255)  # green
sobelY_thresh = (25, 255)  # green
sobelMag_thresh = (50, 225)
sChan_thresh = (100, 255)  # blue
hChan_thresh = (0, 80)
sobelDir_thresh = (0.7, 1.1)
leftLine = Line()
rightLine = Line()
orientx = 'x'
orienty = 'y'
ym_per_pix = 30/720 # meters per pixel in y dimension
xm_per_pix = 3.7/700 # meters per pixel in x dimension


def calibrateCamera():

    # Pass in all calibration images into images list
    imageList = glob.glob(
        '/Users/sumedhinamdar/Documents/GitHub/CarND-Advanced-Lane-Lines/camera_cal/calibration*.jpg')
    nx = 9  # num of corners in x-dir
    ny = 6  # num of corners in y-dir

    objpoints = []  # 3d points in real world space
    imgpoints = []  # 2d points in image plane

    # Generate objpoints - 3D points in real world space
    objp = np.zeros((nx * ny, 3), np.float32)
    objp[:, :2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2)  # [x,y] coords

    # Iterate through all calibration images to find corner points for camera calibration
    for image in imageList:
        # Read in image, convert to grayscale, and find inner corners
        img = mpimg.imread(image)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)

        # Add corner points to arrays if found in image
        if ret == True:
            imgpoints.append(corners)
            objpoints.append(objp)
            # Draw corners found from grayscale image back on original image
            cornersDrawn = cv2.drawChessboardCorners(
                img, (nx, ny), corners, ret)
            # Save the image in a new folder
            mpimg.imsave(image.replace(
                'camera_cal', 'camera_cal_cornersDrawn'), cornersDrawn)

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, gray.shape[::-1], None, None)
    # Read images with detected and drawn corners
    imageList = glob.glob(
        '/Users/sumedhinamdar/Documents/GitHub/CarND-Advanced-Lane-Lines/camera_cal_cornersDrawn/calibration*.jpg')
    # Iterate through images and undistort with mtx and dist from calibration step
    for image in imageList:
        img = mpimg.imread(image)
        undist = cv2.undistort(img, mtx, dist, None, mtx)
        # Save undistorted image in new folder
        mpimg.imsave(image.replace('camera_cal_cornersDrawn',
                                 'camera_cal_undist'), undist)
    return mtx, dist

def colorGradientPipeLine(img):

    # Convert to HLS color space
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    h_channel = hls[:, :, 0]
    l_channel = hls[:, :, 1]
    s_channel = hls[:, :, 2]

    #Delete if no issues: gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Perform x and y sobel gradient threshold
    sobelX = abs_sobel_thresh(s_channel, sobel_kernel, sobelX_thresh, orientx)
    sobelY = abs_sobel_thresh(s_channel, sobel_kernel, sobelY_thresh, orienty)

    # Perform sobel magnitude threshold
    sobelMag = mag_thresh(img, sobel_kernel, sobelMag_thresh)

    # Perform sobel direction threshold
    sobelDir = dir_threshold(img, sobel_kernel, sobelDir_thresh)

    # Perform s-channel color threshold
    sChan_binary = np.zeros_like(s_channel)
    sChan_binary[(s_channel >= sChan_thresh[0])
                 & (s_channel <= sChan_thresh[1])] = 1

    # Perform h-channel color threshold
    hChan_binary = np.zeros_like(h_channel)
    hChan_binary[(h_channel >= hChan_thresh[0])
                 & (h_channel <= hChan_thresh[1])] = 1


    combined = np.zeros_like(s_channel)
    combined[(((sobelX == 1) & (sobelY == 1)) | ((sobelMag == 1) & (
        sobelDir == 1))) | ((sChan_binary == 1) & (hChan_binary == 1))] = 1

    # debug code  - add color to channels
    # combined = np.dstack(( np.zeros_like(sobelX), sobelMag, sChan_binary)) * 255 #RGB image
    return combined

# helper function for thresholding
def abs_sobel_thresh(img, sobel_kernel, thresh, orient):

    # Take the derivative in x or y given orient = 'x' or 'y'
    sobelDir = cv2.Sobel(img, cv2.CV_64F, int(
        orient == 'x'), int(not orient == 'x'), ksize=sobel_kernel)
    # Take the absolute value of the derivative or gradient
    abs_sobelDir = np.absolute(sobelDir)
    # Scale to 8-bit (0 - 255) then convert to type = np.uint8
    scaled_sobelDir = np.uint8(255 * abs_sobelDir / np.max(abs_sobelDir))
    # Create a mask of 1's where the scaled gradient magnitude
    # is > thresh_min and < thresh_max
    scaled_binary = np.zeros_like(scaled_sobelDir)
    scaled_binary[(scaled_sobelDir >= thresh[0])
                  & (scaled_sobelDir <= thresh[1])] = 1
    # Return this mask as your binary_output image
    return scaled_binary

# helper function for thresholding
def mag_thresh(img, sobel_kernel, mgthresh):
    # Apply the following steps to img
    # 1) Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # 2) Take the gradient in x and y separately
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # 3) Calculate the magnitude
    magxy = np.sqrt((sobelx)**2 + (sobely)**2)
    # 4) Scale to 8-bit (0 - 255) and convert to type = np.uint8
    scaled_sobelxy = np.uint8(255 * magxy / np.max(magxy))
    # 5) Create a binary mask where mag thresholds are met
    binary_sobelxy = np.zeros_like(scaled_sobelxy)
    binary_sobelxy[(scaled_sobelxy >= mgthresh[0])
                   & (scaled_sobelxy <= mgthresh[1])] = 1
    # 6) Return this mask as your binary_output image
    return binary_sobelxy

# helper function for thresholding
def dir_threshold(img, sobel_kernel, thresh):

    # Apply the following steps to img
    # 1) Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # 2) Take the gradient in x and y separately
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # 3) Take the absolute value of the x and y gradients
    abs_sobelx = np.absolute(sobelx)
    abs_sobely = np.absolute(sobely)
    # 4) Use np.arctan2(abs_sobely, abs_sobelx) to calculate the direction of the gradient
    atan_sobel = np.arctan2(abs_sobely, abs_sobelx)
    # 5) Create a binary mask where direction thresholds are met
    binary_output = np.zeros_like(atan_sobel)
    binary_output[(atan_sobel >= thresh[0]) & (atan_sobel <= thresh[1])] = 1
    # 6) Return this mask as your binary_output image
    return binary_output

def warpPerspective(image):
    global src
    global dst
    global M
    global Minv
    global img_size

    img_size = (image.shape[1], image.shape[0])
    src = np.float32(
        [[(img_size[0] / 2) - 55, img_size[1] / 2 + 100],
        [((img_size[0] / 6) - 10), img_size[1]],
        [(img_size[0] * 5 / 6) + 60, img_size[1]],
        [(img_size[0] / 2 + 55), img_size[1] / 2 + 100]])
    dst = np.float32(
        [[(img_size[0] / 4), 0],
        [(img_size[0] / 4), img_size[1]],
        [(img_size[0] * 3 / 4), img_size[1]],
        [(img_size[0] * 3 / 4), 0]])
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    return cv2.warpPerspective(image, M, img_size, flags=cv2.INTER_LINEAR)

def find_lane_pixels(binary_warped):
    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[binary_warped.shape[0] // 2:, :], axis=0)
    # Create an output image to draw on and visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0] // 2)
    # argmax provides the indicies of the maximum values in the array
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # HYPERPARAMETERS
    # Choose the number of sliding windows
    nwindows = 9
    # Set the width of the windows +/- margin
    margin = 100
    # Set minimum number of pixels found to recenter window
    minpix = 50

    # Set height of windows - based on nwindows above and image shape
    window_height = np.int(binary_warped.shape[0] // nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()  # tuple of size 2
    nonzeroy = np.array(nonzero[0])  # numpy arrary
    nonzerox = np.array(nonzero[1])

    # Current positions to be updated later for each window in nwindows
    leftx_current = leftx_base
    rightx_current = rightx_base

    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window + 1) * window_height
        win_y_high = binary_warped.shape[0] - window * window_height
        ### TO-DO: Find the four below boundaries of the window ###
        win_xleft_low = leftx_current - margin  # Update this
        win_xleft_high = leftx_current + margin  # Update this
        win_xright_low = rightx_current - margin  # Update this
        win_xright_high = rightx_current + margin  # Update this

        # Draw the windows on the visualization image
        cv2.rectangle(out_img, (win_xleft_low, win_y_low),(win_xleft_high, win_y_high), (0, 255, 0), 2)
        cv2.rectangle(out_img, (win_xright_low, win_y_low),
                      (win_xright_high, win_y_high), (0, 255, 0), 2)

        ### TO-DO: Identify the nonzero pixels in x and y within the window ###
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (
            nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (
            nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)

        ### TO-DO: If you found > minpix pixels, recenter next window ###
        if len(good_left_inds) > minpix:
            leftx_current = int(nonzerox[good_left_inds].mean())
        if len(good_right_inds) > minpix:
            rightx_current = int(nonzerox[good_right_inds].mean())
        ### (`right` or `leftx_current`) on their mean position ###

    # Concatenate the arrays of indices (previously was a list of lists of pixels)
    try:
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)
    except ValueError:
        # Avoids an error if the above is not implemented fully
        pass

    # Extract left and right line pixel positions from image (not index)
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    out_img[lefty, leftx] = [255, 0, 0]
    out_img[righty, rightx] = [0, 0, 255]

    return leftx, lefty, rightx, righty, out_img

def searchExistingLanes(binary_warped):
        # HYPERPARAMETER
        # Choose the width of the margin around the previous polynomial to search
        # The quiz grader expects 100 here, but feel free to tune on your own!
        margin = 100

        # Grab activated pixels
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        ### TO-DO: Set the area of search based on activated x-values ###
        ### within the +/- margin of our polynomial function ###
        ### Hint: consider the window areas for the similarly named variables ###
        ### in the previous quiz, but change the windows to our new search area ###
        left_fit = leftLine.best_fit
        right_fit = rightLine.best_fit

        laneLineLeft = left_fit[0] * nonzeroy**2 + \
            left_fit[1] * nonzeroy + left_fit[2]
        laneLineRight = right_fit[0] * nonzeroy**2 + \
            right_fit[1] * nonzeroy + right_fit[2]
        leftx_low = laneLineLeft - margin
        leftx_high = laneLineLeft + margin
        rightx_low = laneLineRight - margin
        rightx_high = laneLineRight + margin

        left_lane_inds = ((nonzerox > leftx_low) & (
            nonzerox < leftx_high))
        right_lane_inds = ((nonzerox > rightx_low) & (
            nonzerox < rightx_high))

        # Again, extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]

        # Fit new polynomials
        left_fit, right_fit, left_fitx, right_fitx, ploty = fit_poly(
            binary_warped.shape, leftx, lefty, rightx, righty)

        out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255

        out_img[lefty, leftx] = [255, 0, 0]
        out_img[righty, rightx] = [0, 0, 255]
        window_img = np.zeros_like(out_img)
        # Color in left and right line pixels
        out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
        out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

        # Generate a polygon to illustrate the search window area
        # And recast the x and y points into usable format for cv2.fillPoly()
        left_line_window1 = np.array(
            [np.transpose(np.vstack([left_fitx - margin, ploty]))])
        left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx + margin,
                                                                        ploty])))])
        left_line_pts = np.hstack((left_line_window1, left_line_window2))
        right_line_window1 = np.array(
            [np.transpose(np.vstack([right_fitx - margin, ploty]))])
        right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx + margin,
                                                                         ploty])))])
        right_line_pts = np.hstack((right_line_window1, right_line_window2))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(window_img, np.int_([left_line_pts]), (0, 255, 0))
        cv2.fillPoly(window_img, np.int_([right_line_pts]), (0, 255, 0))
        result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)

        return leftx, lefty, rightx, righty, result

def fit_poly(img_shape, leftx, lefty, rightx, righty):
    ### TO-DO: Fit a second order polynomial to each with np.polyfit() ###
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    # Generate x and y values for plotting
    ploty = np.linspace(0, img_shape[0] - 1, img_shape[0])
    ### TO-DO: Calc both polynomials using ploty, left_fit and right_fit ###
    left_fitx = left_fit[0] * ploty**2 + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty**2 + right_fit[1] * ploty + right_fit[2]

    return left_fit, right_fit, left_fitx, right_fitx, ploty

def wasLaneDetected(img_shape, leftx, lefty, rightx, righty):

    left_fit, right_fit, left_fitx, right_fitx, ploty = fit_poly(img_shape, leftx, lefty, rightx, righty)

    minPixDetect = 3000

    if len(leftx)<minPixDetect: # Do not update lane variables if we do not detect min number of pixels
        leftLine.detected = False
    else:
        leftLine.detected = True
        # left_fit = np.polyfit(lefty, leftx, 2)
        leftLine.current_fit = left_fit
        # left_fitx = left_fit[0] * ploty**2 + left_fit[1] * ploty + left_fit[2]
        leftLine.recent_xfitted.append(left_fitx)

    if len(rightx)<minPixDetect:
        rightLine.detected = False
    else:
        rightLine.detected = True
        # right_fit = np.polyfit(righty, rightx, 2)
        rightLine.current_fit = right_fit
        # right_fitx = right_fit[0] * ploty**2 + right_fit[1] * ploty + right_fit[2]
        rightLine.recent_xfitted.append(right_fitx)

    return ploty

def EMAcalc(series, alpha):
    sLen = len(series)-1
    num = 0
    den = 0
    alpha = 2/(sLen + 2)
    for idx in range(sLen + 1):
        num += series[(sLen-idx)]*(1-alpha)**idx
        den += (1-alpha)**idx
    return num/den

def fit_polynomial(binary_warped, undist):

    # ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])

    if leftLine.detected == False or rightLine.detected == False:
        leftx, lefty, rightx, righty, out_img = find_lane_pixels(binary_warped) # Find our lane pixels from scratch using sliding window
        ploty = wasLaneDetected(binary_warped.shape, leftx, lefty, rightx, righty)

    else:
        leftx, lefty, rightx, righty, out_img = searchExistingLanes(binary_warped) # Search +/- margin from last lane fitting
        ploty = wasLaneDetected(binary_warped.shape, leftx, lefty, rightx, righty)

    # Reminder to delete this try if no issues
    # try:
    #update class object variables of left lane line

    # x values of the last n fits of the line
    leftLine.recent_xfitted = leftLine.recent_xfitted[-10:] #average x values of the fitted line over the last n=10 iterations
    leftLine.bestx = EMAcalc(leftLine.recent_xfitted, .1)
    # leftLine.bestx = np.average(leftLine.recent_xfitted, axis=0) # take average x values of last n frames
    leftLine.best_fit = np.polyfit(ploty, leftLine.bestx, 2) #polynomial coefficients averaged over the last n iterations
    leftLine.radius_of_curvature = measure_curvature_real(ploty, leftLine.bestx) #radius of curvature of the line in some units
    ymax_l = np.max(ploty)*ym_per_pix
    current_left_x = leftLine.best_fit[0]*ymax_l**2 + leftLine.best_fit[1]*ymax_l + leftLine.best_fit[2]
    leftLine.line_base_pos = binary_warped.shape[1]//2 - current_left_x #distance in meters of vehicle center from the line
    leftLine.diffs = leftLine.best_fit - leftLine.current_fit #difference in fit coefficients between last and new fits
    leftLine.allx = leftx #x values for detected line pixels
    leftLine.ally = lefty #y values for detected line pixels

    rightLine.recent_xfitted = rightLine.recent_xfitted[-10:]
    rightLine.bestx = EMAcalc(rightLine.recent_xfitted, .1)
    # rightLine.bestx = np.average(rightLine.recent_xfitted, axis=0) # take average x values of last 10 frames
    rightLine.best_fit = np.polyfit(ploty, rightLine.bestx, 2)
    rightLine.radius_of_curvature = measure_curvature_real(ploty, rightLine.bestx)
    ymax_r = np.max(ploty)*ym_per_pix
    current_right_x = rightLine.best_fit[0]*ymax_r**2 + rightLine.best_fit[1]*ymax_r + rightLine.best_fit[2]
    rightLine.line_base_pos = abs(binary_warped.shape[1]//2 - current_right_x)
    rightLine.diffs = rightLine.best_fit - rightLine.current_fit
    rightLine.allx = rightx
    rightLine.ally = righty

    ## Visualization ##

    warp_zero = np.zeros_like(binary_warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([leftLine.bestx, ploty]))],np.int32)
    pts_right = np.array([np.flipud(np.transpose(np.vstack([rightLine.bestx, ploty])))],np.int32)
    pts = np.hstack((pts_left, pts_right))

    # Colors in the lane area region
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

    # Plots the left and right polynomial fit points on the warped perspective photo (out_img)
    cv2.polylines(out_img, pts_left, isClosed=False, color=(255,255,0), thickness=3, lineType=cv2.LINE_AA)
    cv2.polylines(out_img, pts_right, isClosed=False, color=(255,255,0), thickness=3, lineType=cv2.LINE_AA)

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, img_size)
    # Combine the result with the original image
    result = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)

    return out_img, result

def measure_curvature_real(ploty, x_values):

    # Find polynomial coefficients of line (converted to meters from pixels)
    fit_cr = np.polyfit(ploty*ym_per_pix, x_values*xm_per_pix, 2)
    # Define y-value where we want radius of curvature
    # We'll choose the maximum y-value, corresponding to the bottom of the image
    y_eval = np.max(ploty)*ym_per_pix
    curverad = (1+(2*fit_cr[0]*y_eval+fit_cr[1])**2)**(3/2)/abs(2*fit_cr[0])
    return curverad

def addTextResult(result):
    # Add curvature and center offset text into resulting image
    computedRadius = np.mean([leftLine.radius_of_curvature,rightLine.radius_of_curvature]).round(decimals=2)
    text1 = ('Radius of curvature = ' + str(computedRadius) + 'm')
    offset = (rightLine.line_base_pos - leftLine.line_base_pos)/2*xm_per_pix
    text2 = ('Vehicle is ' + str(offset.round(decimals=2)) + 'm left of center')
    cv2.putText(result, text1, (30,60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(result, text2, (30,90), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    return result

def addOverlay(main_image, overlay):
    percent_reduction = 0.25
    resized = cv2.resize(overlay, (int(1280*percent_reduction), int(720*percent_reduction) )) #resize to 256x144
    margin = 40
    x_offset = main_image.shape[1]-resized.shape[1]-margin
    y_offset = margin
    main_image[y_offset:y_offset+resized.shape[0], x_offset:x_offset+resized.shape[1]] = resized
    return main_image
