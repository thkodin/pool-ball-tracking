# A heavily-commented version of this code is available in: /individual_tasks/Stitching/img_stitching_detailed.py
# The above is not a class implementation of stitching, but it does the same things.

import os
import sys
import cv2 as cv
import numpy as np
import imutils


class Stitcher:
    def __init__(self, feature_extractor = 'sift', feature_matcher = 'bf', ratio = 0.8, k = 2, reprojThresh = 4):
        script_path = sys.path[0]
        self.save_path = os.path.normpath(os.path.join(script_path, 'stitcher_results'))
        os.makedirs(self.save_path) if not os.path.exists(self.save_path) else print('[INFO-STITCHING] Save directory found.')

        if k != 2:
            print('[ERROR-STITCHING]: Currently, only k = 2 is supported. Please ensure.')
            os._exit(0)

        self.feature_extractor = feature_extractor
        self.feature_matcher = feature_matcher

        if feature_extractor == 'sift':
            self.descriptor = cv.SIFT_create()
            self.distance_metric = cv.NORM_L2        

        elif feature_extractor == 'surf':
            self.descriptor = cv.SURF_create()
            self.distance_metric = cv.NORM_L2

        elif feature_extractor == 'brisk':
            self.descriptor = cv.BRISK_create()
            self.distance_metric = cv.NORM_HAMMING   

        elif feature_extractor == 'orb':
            self.descriptor = cv.ORB_create()
            self.distance_metric = cv.NORM_HAMMING
        
        crossCheck = True if feature_matcher == 'bf' else False
        self.matcher = cv.BFMatcher_create(self.distance_metric, crossCheck = crossCheck)

        print('[INFO-STITCHING] Selected feature extractor ID {} ({}).'
               .format(self.descriptor.descriptorType(), feature_extractor.upper()))

        print('[INFO-STITCHING] Selected feature matcher {}, crossCheck set to {}.'
               .format(feature_matcher.upper(), crossCheck))

        self.reprojThresh = reprojThresh
        self.ratio = ratio         
        self.k = k 


    # Member function to stitch two images together.
    def stitch(self, frames):
        frame_L, frame_R = frames                        
        frame_LGray = cv.cvtColor(frame_L, cv.COLOR_BGR2GRAY)                      
        frame_RGray = cv.cvtColor(frame_R, cv.COLOR_BGR2GRAY)

        hR, wR = frame_R.shape[:2]
        hL, wL = frame_L.shape[:2]
        print('[INFO-STITCHING] Right Image Height: {}, Image Width: {}'.format(hR, wR))
        print('[INFO-STITCHING] Left Image Height: {}, Image Width: {}'.format(hL, wL))

        kptsL, featuresL = self.detectAndDescribe(frame_LGray)
        kptsR, featuresR = self.detectAndDescribe(frame_RGray)

        if self.feature_matcher == 'bf':
            matches = self.matchKeyPointsBF(featuresR, featuresL)
            matches_vis = cv.drawMatches(frame_R, kptsR, frame_L, kptsL, matches[:100], None,
                                        flags = cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

        else:
            matches = self.matchKeyPointsKNN(featuresR, featuresL)
            matches_vis = cv.drawMatches(frame_R, kptsR, frame_L, kptsL, np.random.choice(matches, 100), None,
                                        flags = cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS) 

        cv.imwrite(os.path.join(self.save_path, 'matched_kpts.jpg'), matches_vis)
        #cv.imshow('Matched Features', matches_vis)
        cv.waitKey(1)

        H, _ = self.getHomography(kptsR, kptsL, matches)

        stitched_width = frame_R.shape[1] + frame_L.shape[1]
        stitched_height = frame_R.shape[0] + frame_L.shape[0]

        stitched_result = cv.warpPerspective(frame_R, H, (stitched_width, stitched_height))
        cv.imwrite(os.path.join(self.save_path, 'warped.jpg'), stitched_result) 

        stitched_result[0 : frame_L.shape[0], 0 : frame_L.shape[1]] = frame_L
        cv.imwrite(os.path.join(self.save_path, 'stitched.jpg'), stitched_result)

        print('[INFO-STITCHING] Stitching successful. Refining result...')
        stitched_result_max, stitched_result_min, stitch_maxrect, stitch_minrect = self.refineStitch(stitched_result)

        cv.imwrite(os.path.join(self.save_path, 'stitched_maxFit.jpg'), stitched_result_max)
        cv.imwrite(os.path.join(self.save_path, 'stitched_minFit.jpg'), stitched_result_min)
        cv.imshow('Stitched Result (Final)', stitched_result_max)
        cv.waitKey(0)

        print('[SUCCESS-STITCHING] Stitch successful. Results saved to folder "stitcher_results".')
        print('----------------------------------------------------------------------------------')

        # Clean up.
        cv.destroyWindow('Stitched Result (Final)')
        #cv.destroyWindow('Matched Features')
        
        return stitched_result_max, stitched_result_min, stitch_maxrect, stitch_minrect, H       


    def detectAndDescribe(self, image):     
        (kpts, features) = self.descriptor.detectAndCompute(image, None)
        return (kpts, features)


    def matchKeyPointsBF(self, featuresR, featuresL):
        best_matches = self.matcher.match(featuresR, featuresL)
        rawMatches = sorted(best_matches, key = lambda match:match.distance)
        print('[INFO-STITCHING] Raw Matches (Brute Force):', len(rawMatches))
        return rawMatches


    def matchKeyPointsKNN(self, featuresR, featuresL):
        rawMatches = self.matcher.knnMatch(featuresR, featuresL, self.k)
        print('[INFO-STITCHING] Raw Matches (KNN):', len(rawMatches))
        trueMatches = []    

        for m, n in rawMatches:
            if m.distance < n.distance * self.ratio:
                trueMatches.append(m)
        return trueMatches


    def getHomography(self, kptsR, kptsL, matches):
        kptsR = np.float32([kp.pt for kp in kptsR])
        kptsL = np.float32([kp.pt for kp in kptsL])

        if len(matches) > 4:
            ptsR = np.float32([kptsR[m.queryIdx] for m in matches])
            ptsL = np.float32([kptsL[m.trainIdx] for m in matches])
            H, status = cv.findHomography(ptsR, ptsL, cv.RANSAC, self.reprojThresh)
            return H, status

        else:
            print('Unable to compute homography. Please adjust your camera setup and re-initialize.')
            os._exit(0)

    # This function not in img_stitching_detailed.py, but all of what it does is present in the
    # main body for that script. It's made into a function for compactness here. There are a few
    # modifications in order to make the stitching results a bit more robust for our application.
    def refineStitch(self, stitched_image):
        erodil_krnl = np.ones((5, 5), np.uint8)     # kernel for erosion and dilation

        gray = cv.cvtColor(stitched_image, cv.COLOR_BGR2GRAY)
        _, thresh = cv.threshold(gray, 0, 255, cv.THRESH_BINARY)

        cnts = cv.findContours(thresh.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        c = max(cnts, key = cv.contourArea)
        (xmax, ymax, wmax, hmax) = cv.boundingRect(c)

        stitched_image_max = stitched_image[ymax : ymax + hmax, xmax : xmax + wmax]
        stitched_maxshape = stitched_image_max.shape[:2]
        stitch_maxrect = (0, 0, wmax, hmax)
        
        stitched_image_bordered = cv.copyMakeBorder(stitched_image_max.copy(), 10, 10, 10, 10, cv.BORDER_CONSTANT, (0, 0, 0))
        gray = cv.cvtColor(stitched_image_bordered, cv.COLOR_BGR2GRAY)

        _, thresh_lossless = cv.threshold(gray, 0, 255, cv.THRESH_BINARY)

        # Series of dilations and erosions to fill any holes due to black objects.
        thresh_lossless = cv.dilate(thresh_lossless, erodil_krnl, iterations = 2, borderType = cv.BORDER_CONSTANT)
        thresh_lossless = cv.erode(thresh_lossless, erodil_krnl, iterations = 2, borderType = cv.BORDER_CONSTANT)

        cnts = cv.findContours(thresh_lossless.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        c = max(cnts, key = cv.contourArea)

        mask_lossless = np.zeros(thresh_lossless.shape, dtype = 'uint8')         
        (x, y, w, h) = cv.boundingRect(c)
        cv.rectangle(mask_lossless, (x, y), (x + w, y + h), 255, -1)

        mask_minloss = mask_lossless.copy()   
        mask_sub = mask_lossless.copy()        

        while cv.countNonZero(mask_sub) > 0:
            mask_minloss = cv.erode(mask_minloss, None)             
            mask_sub = cv.subtract(mask_minloss, thresh_lossless)   

        cnts = cv.findContours(mask_minloss.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        c = max(cnts, key = cv.contourArea)
        (xmin, ymin, wmin, hmin) = cv.boundingRect(c)

        stitched_image_min = stitched_image_bordered[ymin : ymin + hmin, xmin : xmin + wmin]
        stitched_minshape = stitched_image_min.shape[:2]

        deltaX = stitched_maxshape[1] - stitched_minshape[1] 
        deltaY = stitched_maxshape[0] - stitched_minshape[0]
        stitch_minrect = (deltaX, deltaY, wmin, hmin)

        return stitched_image_max, stitched_image_min, stitch_maxrect, stitch_minrect

# TESTING...
# stitcher = Stitcher()

# frame_L = cv.imread('fotoL.jpg')
# frame_R = cv.imread('fotoR.jpg')  

# result = stitcher.stitch([frame_L, frame_R])
# cv.imshow('Stitched', result); cv.waitKey(0)
# cv.destroyWindow('Stitched')