import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
import os
from moviepy.editor import VideoFileClip
from line import Line
import helpers

#   This function will take in a raw road image and return an overlayed image with the lane area filled in green


def process_image(image):
    undist = cv2.undistort(image, mtx, dist, None, mtx)  # Undistort raw Image
    # Combine color and gradient thresholds to produce binary image from input
    binary_output = helpers.colorGradientPipeLine(undist)
    # Perspective transform on both binary image and original image (debug)
    warped_binary = helpers.warpPerspective(binary_output)
    # Find lane lines through sliding windows and polyfit
    slideWindow_img, result = helpers.fit_polynomial(warped_binary, undist)
    resultwText = helpers.addTextResult(result)  # Put text on resulting image
    final_result = helpers.addOverlay(resultwText, slideWindow_img)
    return final_result


# Calibrate camera with sample images
mtx, dist = helpers.calibrateCamera()

# Create line objects (leftline and rightLine)
helpers.leftLine = Line()
helpers.rightLine = Line()

# Video processing only
video = '/Users/sumedhinamdar/Documents/GitHub/CarND-Advanced-Lane-Lines/challenge_video.mp4'
clip1 = VideoFileClip(video)
# NOTE: this function expects color images!!
white_clip = clip1.fl_image(process_image)
white_output = video.replace('.mp4', '_lanesFound.mp4')
white_clip.write_videofile(white_output, audio=False)

# # Image processing only
# # Pass in test images to imageList
# imageList = glob.glob(
#     '/Users/sumedhinamdar/Documents/GitHub/CarND-Advanced-Lane-Lines/test_images/*.jpg')
#
# for image in imageList:
#     img = mpimg.imread(image)
#     # print('image: ' + image.strip('/Users/sumedhinamdar/Documents/GitHub/CarND-Advanced-Lane-Lines/test_images/'))
#     # Step 1: Undistort images using mtx and dist coefficients from calibrateCamera
#     undist = cv2.undistort(img, mtx, dist, None, mtx)
#     mpimg.imsave(image.replace('test_images', 'output_images').replace(
#         '.jpg', '_undist.jpg'), undist)
#
#     # Step 2: Use color/gradient thresholds to produce binary image
#     binary_output = helpers.colorGradientPipeLine(undist)
#     mpimg.imsave(image.replace('test_images', 'output_images').replace(
#         '.jpg', '_binaryOut.jpg'), binary_output, cmap='gray')
#
#     # Step 3: Perspective transform on both binary image and original image (debug)
#     warped_binary = helpers.warpPerspective(binary_output)
#     #cv2.polylines(warped_binary, [dst.astype(int).reshape((-1, 1, 2))], True, (255, 0, 0), 6)
#     mpimg.imsave(image.replace('test_images', 'output_images').replace(
#         '.jpg', '_warpBinary.jpg'), warped_binary, cmap='gray')
#
#     #debug
#     warped_original = helpers.warpPerspective(undist)
#     cv2.polylines(warped_original, [helpers.dst.astype(int).reshape((-1, 1, 2))], True, (255, 0, 0), 6)
#     mpimg.imsave(image.replace('test_images', 'output_images').replace(
#         '.jpg', '_warpOG.jpg'), warped_original)
#     #debug
#     # cv2.polylines(imgCopy, [helpers.src.astype(int).reshape((-1, 1, 2))], True, (255, 0, 0), 6)
#     # mpimg.imsave(image.replace('test_images', 'output_images').replace(
#     #     '.jpg', '_unwarpedOG.jpg'), imgCopy, cmap='gray')
#
#     # Step 4: Find lane lines through sliding windows and polyfit
#     helpers.leftLine = Line()
#     helpers.rightLine = Line()
#     slideWindow_img, result = helpers.fit_polynomial(warped_binary, undist)
#     mpimg.imsave(image.replace('test_images', 'output_images').replace(
#         '.jpg', '_slideWindow.jpg'), slideWindow_img)
#     # plt.imshow(slideWindow_img)
#     # plt.savefig(image.replace('test_images', 'output_images').replace(
#     #     '.jpg', '_slideWindow.jpg'))
#     # plt.clf()
#
#
#     # put text on resulting image
#     resultwText = helpers.addTextResult(result)
#     mpimg.imsave(image.replace('test_images', 'output_images').replace(
#         '.jpg', '_resultwText.jpg'), resultwText)
#
#     # overlay sliding windows onto final image
#     final_result = helpers.addOverlay(resultwText, slideWindow_img)
#     mpimg.imsave(image.replace('test_images', 'output_images').replace(
#         '.jpg', '_finalResult.jpg'), final_result)


# #
# # #
# # #
# # # # #test code
# # # # # Plot the result
# # # # f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
# # # # f.tight_layout()
# # # #
# # # # ax1.imshow(img)
# # # # ax1.set_title('Original Image', fontsize=40)
# # # #
# # # # ax2.imshow(binary_output, cmap='gray')
# # # # ax2.set_title('Binary Result', fontsize=40)
# # # # plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
# # # # plt.show()
# # #
# # # #debug
