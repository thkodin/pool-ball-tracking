# Ensure image sizes are the same for best results.
# Framework used: 2D Features Framework by OpenCV.
# https://docs.opencv.org/3.4/da/d9b/group__features2d.html

import os
import sys
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import imutils

script_path = sys.path[0]
save_path = os.path.normpath(os.path.join(script_path, "stitched_results"))
os.makedirs(save_path) if not os.path.exists(save_path) else print("[INFO-STITCHING] Save directory found.")

# Decide algorithm for feature detection and descripton. "sift", "surf", "orb", "brisk" are some options.
# "descriptor" is a Feature2D class object, described in OpenCV documentation here:
# https://docs.opencv.org/3.4/d0/d13/classcv_1_1Feature2D.html

# SIFT abd SURF are no longer patented. cv.xfeatures2D.SIFT_create() is no longer correct.
# Use cv.SIFT_create() in OpenCV versions > 4.4.

feature_matcher = "bf"      # brute force or KNN                  
feature_extractor = "sift"  # one of "sift", "surf", "brisk", "orb"

ratio = 0.8         # (KNN) generally taken around 0.7 to 0.8
k = 2               # (KNN) shouldn't be more than 2 for now
reprojThresh = 3    # (HOMOGRAPHY) minimum reprojection error acceptable; needed since RANSAC is random

if k != 2:
    print("[ERROR-STITCHING]: Currently, only k = 2 is supported. Please ensure.")
    os._exit(0)

# Construct appropriate feature extraction algorithm, and set recommended distance metric for that algo.
if feature_extractor == "sift":
    descriptor = cv.SIFT_create()
    distance_metric = cv.NORM_L2        # Euclidean distance to match descriptors
elif feature_extractor == "surf":
    descriptor = cv.SURF_create()
    distance_metric = cv.NORM_L2

elif feature_extractor == "brisk":
    descriptor = cv.BRISK_create()
    distance_metric = cv.NORM_HAMMING   # Hamming distance to match descriptors
elif feature_extractor == "orb":
    descriptor = cv.ORB_create()
    distance_metric = cv.NORM_HAMMING

print("[INFO-STITCHING] Selected feature extractor ID {} ({})."
      .format(descriptor.descriptorType(), feature_extractor.upper()))

# Create matcher object. Brute Force matchers are used for implementing both 'brute force' and 'KNN' matchers, where BF requires
# crossCheck parameter of constructor to be True and KNN requires it to be False. See below...
# https://docs.opencv.org/3.4/d3/da1/classcv_1_1BFMatcher.html
# Note that cv.BFMatcher() is used as a matcher object constructor in some references. That's considered obsolete by OpenCV, 
# who recommend the constructor used below. First argument to this is the distance metric used (euclidean for SIFT and SURF,
# hamming for ORB and BRISK). The second is a boolean crossCheck which, when set, ensures that the distance between descriptors 
# of a pair of features (f1, f2) is such that f1 is the CLOSEST match to f2 AND f2 is the CLOSEST match to f1. That is, if f1
# is the closest match to f2 but f2 is the closest match to some other feature, then (f1, f2) is discarded! This is necessary
# for brute force matching and is considered very robust. However, it may lead to insufficienct correspondence. In that case,
# we can use KNN for feature matching by clearing crossCheck and defining 'k' number of closest matches to be considered valid.
# However, then we must test them for robustness using the ratio method proposed by David Lowe (author of SIFT) i.e. the distances
# must be within a certain ratio of each other.
 
crossCheck = True if feature_matcher == "bf" else False
print("[INFO-STITCHING] Selected feature matcher {}, crossCheck set to {}."
       .format(feature_matcher.upper(), crossCheck))

# Construct the matcher object with appropriate arguments.
matcher = cv.BFMatcher_create(distance_metric, crossCheck = crossCheck)


# Compute key points and feature descriptors using an specific method. Defaults to SIFT.
def detectAndDescribe(descriptor, image):     
    # Get keypoints and descriptors for those keypoints using method of Feature2D class object "detectAndCompute(image, mask)".
    # Mask is useful if you have a specific area to focus stitching, or bad features in some part of the image. It can also help 
    # identify the overlapping region in two frames if user inputs it manually using e.g. selectROI.
    # Further, note that 'features' are actually feature descriptors here.
    (kpts, features) = descriptor.detectAndCompute(image, None)
    return (kpts, features)


# BF and knnMatch() described here:
# https://docs.opencv.org/3.4/db/d39/classcv_1_1DescriptorMatcher.html#a38f18a66b3498fa5c960e29ed0056d44
def matchKeyPointsBF(matcher, featuresR, featuresL):
    # Match descriptors with brute force. This will return only the most robust matches.
    best_matches = matcher.match(featuresR, featuresL)
    
    # Sort the features in order of distance.
    # The points with small distance (more similarity) are ordered first in the vector
    rawMatches = sorted(best_matches, key = lambda match:match.distance)
    print("[INFO-STITCHING] Raw Matches (Brute Force):", len(rawMatches))
    return rawMatches


def matchKeyPointsKNN(matcher, featuresR, featuresL, k, ratio):
    # For each feature in the train image (left view), match k nearest descriptors in the query image (right view). 
    # Each raw match therefore has k elements, which must pass the ratio test before being considered valid.
    rawMatches = matcher.knnMatch(featuresR, featuresL, k)
    print("[INFO-STITCHING] Raw Matches (KNN):", len(rawMatches))
    trueMatches = []    

    for m, n in rawMatches:
        if m.distance < n.distance * ratio:  # ratio test for k = 2 ONLY
            trueMatches.append(m)

    return trueMatches

    # FOR K > 2...
    # RATIO TEST IMPLEMENTED CURRENTLY ONLY FOR K = 2!
    # For each raw match...
    # for m in rawMatches:        
    #     # Apply ratio test.
    #     i = 1
    #     while i <= k:
    #         if m[0].distance < m[i].distance * ratio: matches.append(m[0])
    #         i += 1

    # return matches

# Compute a homography that projects right image (query) onto left image (train). We'll use RANSAC to estimate it.
def getHomography(kptsR, kptsL, matches, reprojThresh):
    # Convert the keypoints to numpy arrays. Keypoints have the attribute '.pt' for this.
    kptsR = np.float32([kp.pt for kp in kptsR])
    kptsL = np.float32([kp.pt for kp in kptsL])
    
    # Minimum 4 matches necessary to get homography of 2D images! 4 matches yield 8 equations which
    # matches the number of unknowns (not counting 9th) in the original equation. Obviously, the 
    # more matches, the better the results.
    if len(matches) > 4:
        # Construct the two sets of points to compute the homography between. Matches are corresponding pairs of 
        # keypoints. In OpenCV, each match has a queryIdx and trainIdx attribute that describes the matched pair's 
        # keypoint indices (pixel locs) image-wise. That is, the queryIdx tells the matched feature (keypoint) 
        # index in the keypoints of query_image. While trainIdx tells the same for train_image. In case of KNN,  
        # these are indices of k matches, not just one.
        ptsR = np.float32([kptsR[m.queryIdx] for m in matches])
        ptsL = np.float32([kptsL[m.trainIdx] for m in matches])
        
        # Estimate the homography between the sets of points.
        H, status = cv.findHomography(ptsR, ptsL, cv.RANSAC, reprojThresh)
        return H, status

    else:
        print("Unable to compute homography. Please adjust your camera setup and re-initialize.")
        os._exit(0)

# Make sure that the right frame image is the one that will be transformed! This constraint is internal
# to this framework for OpenCV. The functions require a train_image and a query_image. The train_image
# IS NOT TRANSFORMED while the query_image IS TRANSFORMED/STITCHED to the train_image! You can change which
# side (left or right) serves as which type of image, but you'd have to change the order in the OpenCV funcs
# as well, so just don't for simplicity. In general, OpenCV functions for this framework always put the
# query_image BEFORE the train_image in their arguments. 

# Left view... or train_img, as OpenCV calls it
frame_L = cv.imread("debug_stitcherL.jpg")                        
frame_LGray = cv.cvtColor(frame_L, cv.COLOR_BGR2GRAY)

# Right view... or query_image, as OpenCV calls it
frame_R = cv.imread("debug_stitcherR.jpg")                        
frame_RGray = cv.cvtColor(frame_R, cv.COLOR_BGR2GRAY)

# Get height and width of both frames.
hR, wR = frame_R.shape[:2]
print("[INFO-STITCHING] Right Image Height: {}, Image Width: {}".format(hR, wR))

hL, wL = frame_L.shape[:2]
print("[INFO-STITCHING] Left Image Height: {}, Image Width: {}".format(hL, wL))

# DEBUGGING
cv.imshow("Image to stitch to (left camera)", frame_L)     # will not be transformed
cv.imshow("Image to be stitched (right camera)", frame_R)  # will be transformed
cv.waitKey(1)

# Call feature detector and describer
kptsL, featuresL = detectAndDescribe(descriptor, frame_LGray)
kptsR, featuresR = detectAndDescribe(descriptor, frame_RGray)

# Visualize keypoints by calling the 2DFeatures class method "drawKeypoints". 3rd argument is the 'outImg'
# where these keypoints are drawn. We set to None sinc we are assigning fresh memory to these.
det_kptsL = cv.drawKeypoints(frame_LGray, kptsL, None, color = (150, 150, 255))
det_kptsR = cv.drawKeypoints(frame_RGray, kptsR, None, color = (150, 150, 255))

cv.imshow("Keypoints (left image)", det_kptsL)
cv.imshow("Keypoints (right image)", det_kptsR)
cv.waitKey(1)

cv.imwrite(os.path.join(save_path, "det_kptsL.jpg"), det_kptsL)
cv.imwrite(os.path.join(save_path, "det_kptsR.jpg"), det_kptsR)

# Draw at most a 100 matches for visualization purposes. In case of BF, pick the best 100 (i.e. the first 100)
# since we sorted them). In case of KNN, choose them randomly. 5th argument is outImg, we leave it at None for
# same reason as earlier. The flag tells it some further parameters for drawing.
if feature_matcher == "bf":
    matches = matchKeyPointsBF(matcher, featuresR, featuresL)
    matches_vis = cv.drawMatches(frame_R, kptsR, frame_L, kptsL, matches[:100], None,
                                 flags = cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

else:
    matches = matchKeyPointsKNN(matcher, featuresR, featuresL, k = k, ratio = ratio)
    matches_vis = cv.drawMatches(frame_R, kptsR, frame_L, kptsL, np.random.choice(matches, 100), None,
                                 flags = cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS) 

cv.imshow("Matched Features", matches_vis)
cv.waitKey(1); cv.imwrite(os.path.join(save_path, "matched_kpts.jpg"), matches_vis)

# Get the homography between matched keypoints.
H, status = getHomography(kptsR, kptsL, matches, reprojThresh = reprojThresh)

# Create a canvas big enough to hold the stitched results. Just add the shapes of the two frames!
stitched_width = frame_R.shape[1] + frame_L.shape[1]
stitched_height = frame_R.shape[0] + frame_L.shape[0]

# Apply perspective transform ONLY TO QUERY_IMAGE. In this case, it is the right side frame.
stitched_result = cv.warpPerspective(frame_R, H, (stitched_width, stitched_height))
cv.imshow("Warped Perspective (Right Frame)", stitched_result)
cv.waitKey(1); cv.imwrite(os.path.join(save_path, "warped.jpg"), stitched_result) 

# Fit the train_image (left frame in this case) into the top left corner of the canvas.
stitched_result[0 : frame_L.shape[0], 0:frame_L.shape[1]] = frame_L
cv.imshow("Stitched (Left + Right)", stitched_result)
cv.waitKey(0); cv.imwrite(os.path.join(save_path, "stitched.jpg"), stitched_result)

# Once images stitched, we now remove the excessive part of the canvas.
print("[INFO-STITCHING] Stitching Successful. Refining result...")
# transform the panorama image to grayscale and threshold it 
gray = cv.cvtColor(stitched_result, cv.COLOR_BGR2GRAY)
_, thresh = cv.threshold(gray, 0, 255, cv.THRESH_BINARY)

# Finds contours from the binary image. APPROX_SIMPLE helps extract polygon shapes (such as rectangles).
# RETR_EXTERNAL only extracts the external bounding contours in the image.
cnts = cv.findContours(thresh.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)

# Extract the contour with maximum area.
c = max(cnts, key = cv.contourArea)

# Draw bbox around the extracted contour.
(x, y, w, h) = cv.boundingRect(c)
stitched_result = stitched_result[y : y + h, x : x + w]
cv.imwrite(os.path.join(save_path, "stitched_maxFit.jpg"), stitched_result)

# Now we have the lossless refinement. We can choose to further crop it (will lose data)
# to remove any black edges that are left. Following Adrian's hack given here:
# https://www.pyimagesearch.com/2018/12/17/image-stitching-with-opencv-and-python/

# Pad the border in all directions by 10 black pixels to ensure the contours are properly established.
stitched_result = cv.copyMakeBorder(stitched_result, 10, 10, 10, 10, cv.BORDER_CONSTANT, (0, 0, 0))
gray = cv.cvtColor(stitched_result, cv.COLOR_BGR2GRAY)

# thresh contains the black regions in the lossless results.
_, thresh_lossless = cv.threshold(gray, 0, 255, cv.THRESH_BINARY)

# Apply contour extraction again as we have padded by 10. Same parameters. This contour will
# represent the lossless stitched result which we will further operate on.
cnts = cv.findContours(thresh_lossless.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
c = max(cnts, key = cv.contourArea)

# Create a rectangular mask over the largest bounding contour over the stitched results. This mask
# covers our stitched image perfectly, and is thus the best-fit rectangle.
mask_lossless = np.zeros(thresh_lossless.shape, dtype = "uint8")         
(x, y, w, h) = cv.boundingRect(c)
cv.rectangle(mask_lossless, (x, y), (x + w, y + h), 255, -1)

mask_minloss = mask_lossless.copy()   # will be our desired mask; has no black regions and minimzes loss
mask_sub = mask_lossless.copy()       # to compare threshold (non-rectangular lossless) with eroded mask (rect loss) 

# This is best explained with a figure. Basically, erode the lossless rectangular mask until it fits completely
# inside the lossless threshold. The decision is made by subtracting the eroded version of the lossless mask from
# the lossless threshold (which remains constant).
while cv.countNonZero(mask_sub) > 0:
    mask_minloss = cv.erode(mask_minloss, None)             # thin the mask
    mask_sub = cv.subtract(mask_minloss, thresh_lossless)   # subtract the thinned mask from lossless threshold

# Get the contours once final time inside the eroded mask, mask_minloss.
cnts = cv.findContours(mask_minloss.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
c = max(cnts, key = cv.contourArea)
(x, y, w, h) = cv.boundingRect(c)

# Final crop and we're done. Hell yeah.
stitched_result = stitched_result[y : y + h, x : x + w]
# Note that this does not work every time, owing to it being a hack. You might need to change some things
# up to get the proper minFit.
cv.imwrite(os.path.join(save_path, "stitched_minFit.jpg"), stitched_result)