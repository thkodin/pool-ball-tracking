########################## EXPLANATORY CODE BLOCK FOR YOLO ON AN IMAGE ###############################

import os           # used in offline mode path manipulations
import sys          # used in online mode for specifying recording directory
import threading    # used to implemented multi-threading (lag reduction)
import cv2 as cv    # used for obvious reasons --- the network functions are also from here!
import numpy as np
import time

# FOR SINGLE IMAGE. We'll be following pyimagesearch in their excellent tutorial here:
# https://www.pyimagesearch.com/2018/11/12/yolo-object-detection-with-opencv/

# USER ADJUSTABLE PATHS --- ADJUST ACCORDING TO YOUR DIRECTORY LOCATIONS!!!
cudaDir = 'C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.5/bin'
darknetDir = 'D:/YoloV4_DarkNet/darknet'
os.add_dll_directory(cudaDir)        # dependency of darknet 
os.add_dll_directory(darknetDir)     # darknet itself

### YOLO REQUISITES --- DEFINE THESE AT THE TOP OF YOUR SCRIPT ###
# Path to the names for the YOLO network to use.
namesPath = os.path.join(darknetDir, 'cfg/coco.names').replace(os.sep, '/')
print("(Names)\t\t", namesPath)
# Path to the configuration of the YOLO network for use.
# CFG DICTATES NUMBER OF CLASSES!!! VERY IMPORTANT IF YOU WANT TO REDUCE DETECTABLE CLASSES!!!
# For the things you need to change if you only wanted to detect cars for instance:
# https://stackoverflow.com/questions/57898577/how-to-reduce-number-of-classes-in-yolov3-files
cfgPath = os.path.join(darknetDir, 'cfg/yolov4.cfg').replace(os.sep, '/')
print("(Configs)\t", cfgPath)
# Path to the weights (either custom or pre-trained) for use.
weightsPath = os.path.join(darknetDir, 'yolov4.weights').replace(os.sep, '/')
print("(Weights)\t", weightsPath)

# Excellent explanation of random.seed at: 
# https://stackoverflow.com/questions/21494489/what-does-numpy-random-seed0-do
# For now, just know that it causes the random numbers to be generated from the same starting point,
# so every script run gets you the same set of random numbers. This is why the seed default is 
# .seed(), because then it uses system time which is always changing.
np.random.seed(100)
# Here, the first size dimension (rows) is equal to the number of unique labels in your data class.
# Read the .names file like in the tutorial link for variable no. of class names, but make sure your
# .cfg is ready to handle that many classes as well using the link above cfgPath.
# For our usage of YOLO, we will only need one color so no need to go randoming stuff.
labels = np.loadtxt(namesPath, dtype = str, delimiter = ',')
colors = np.random.randint(0, 255, size = (len(labels), 3), dtype = np.uint8)

# Load the detector --- BEFORE READING ANYTHING
print("Loading YOLO from", darknetDir, end = '...\n')
yolo = cv.dnn.readNetFromDarknet(cfgPath, weightsPath)
yolo.setPreferableBackend(cv.dnn.DNN_BACKEND_CUDA)
yolo.setPreferableTarget(cv.dnn.DNN_TARGET_CUDA)


layer_names = yolo.getLayerNames()  # this gets every single YOLO layer's name

# We need the output layers' names. These names are called yolo layers in the layer_names variable.
# YOLOv4 has 3 output layers (with the final one having no forward connection at all, and the first 2 
# being intermediary outputs) in the last 11 layers of the entire network. Since our model has not had
# a forward pass yet, the output layers are not yet connected to the layer in front of them! Thus, we
# can get the positions of the output layers by getting the positions (indices in layer_names) of the 
# layers in front of them and subtracting one. This even works for the last yolo layer which is 
# technically the last element of layer_names. Most probably because the .getUcLayers gets the output
# layer and index and adds one to it, which is the opposite of what we are now doing :)

# The method must have got updated, because the pyimagesearch blog has i[0] - 1... so maybe 
# getUcLayers gave a list of a list back then, but now it just gives a list of indices so just use 
# i - 1 now. Throws an error otherwise.
layer_names = [layer_names[i - 1] for i in yolo.getUnconnectedOutLayers()]  # only output layers


# It's best to implement this as a callable function for expandability.
# In order: input image, data file, config file, weights file, minimum confidence for validation of
# detection, and threshold for non-maxima suppression (NMS). Note that this minConf is the equivalent 
# of thresh in darknet_video.py's arguments. NMS is necessary to remove multiple detections of the 
# same object. It ensures that bboxes with high overlap are ignored!
def yolo_on_frame(frame, h, w, labels, minConf, NMSThresh):
    # Pre-processing the incoming frame for our network!
    # What's a blob? https://www.pyimagesearch.com/2017/11/06/deep-learning-opencvs-blobfromimage-works/?_ga=2.249537473.712614432.1639429511-463802896.1639174332
    # The blob is essentially the frame after pre-processing (mean subtraction etc). Definitely view 
    # the link above for a very comprehensive look on it.
    # image, scaling factor for network, expected image size for network, swap red blue channels for
    # network if it uses RGB ordering for mean subtraction, etc etc...

    # View .cfg file for required input size!
    blob = cv.dnn.blobFromImage(frame, 1/255.0, (608, 608), swapRB = True, crop = False)
    
    # With the blob, perform one forward pass through the network. Single shot's the name after all!
    # We can get all the YOLO we need from this shot.
    yolo.setInput(blob)
    start = time.time()
    layer_outputs = yolo.forward(layer_names)  # only gets the results for layer_names (outputs)
    end = time.time()
    print("...forward pass time (seconds):", end - start)
    
    bboxes = []      # list of bboxes for all detected objects
    confs = []       # list of confidences/probabilities for all detections
    IDs = []         # list of class names/label IDs(YOLO gets both class name and confidence printed)
    
    # For each YOLO output layer...
    for output in layer_outputs:
        # For each detection in that layer...
        for det in output:
            # det's first 4 values are the bbox (x, y) center and then the bbox width, height.
            # 5th value and onwards is the probability score for each of the classes included in
            # the .names file. For the default yolov4.cfg, this is a list of 80 elements. Each
            # element represents the probability that this detection is one of the 80 COCO classes.
            # It is possible to have a detection with multiple scores (a common contention is between
            # cars and trucks that have similar features), in which case the one with highest score
            # is extracted. Since list[5:] starts reading from the 5th element as the 0th index
            # of the list (i.e., while list[5] has index 5 for 5th element, list[5:] starts reading
            # FROM the 5th element, so it calls it index 0), the index corresponds exactly to the class
            # to which the labels belong. The .cfg file must be changed for customizing YOLO like this.
            score = det[5:]
            ID = np.argmax(score)          # get label ID of det
            conf = score[ID]               # get confidence of det
            
            if conf > minConf:
                # YOLO gets you the bbox normalized w.r.t 608 x 608 image size it takes. Thus, if an
                # object was detemrined at the center, it would have (0.5, 0.5, width, height) kind
                # of format where 1st argument is along x-axis (left-right) and 2nd argument is along
                # y-axis (up-down) and width < 1, and so is height. Thus the result must be scaled
                # back to the actual image's coordinates! Further note the difference between x and y
                # in images and matrices: in matrices, the x represents rows and y represents columns
                # by convention. In images, these roles are reversed! A movement along the traditional
                # matrix row is actually going up-down in the image, and is thus called y in images.
                bbox = det[0:4] * np.array([w, h, w, h])
                # Above data is float, so we convert to integers first. Can also use astype, but not 
                # int(bbox) since that method is only applicable to scalar values, not matrices.
                bbox = bbox.astype("int")       
                
                bbox[0] = int(bbox[0] - bbox[2]/2)   # get to leftmost edge of bbox
                bbox[1] = int(bbox[1] - bbox[3]/2)   # get to the top edge of box
                                                     # net effect: reached top-left corner of box
                
                bboxes.append(bbox)                  # store the bbox info for the det
                confs.append(float(conf))            # store the confidence value of the det
                IDs.append(ID)                       # store the label/class of the det
    
    # Apply Non-Maxima Suppression (NMS); YOLO doesn't do it by default. This is to remove the boxes
    # representing the same object. You can just call this removing duplicate detections.
    bbox_idxs_to_keep = cv.dnn.NMSBoxes(bboxes, confs, minConf, NMSThresh)
    
    # This generally won't happen, but on the very small chance it might ---
    # Ensure that a bbox actually is present post-NMS...
    if len(bbox_idxs_to_keep) > 0:
        # Go over all the bboxes to keep. We don't bother deleting them for computational reasons.
        # The list is going to be replaced next frame anyway, why waste time?
        for i in bbox_idxs_to_keep.flatten():
            # Get the bbox coordinates for ease of use in drawing.
            (bx, by, bw, bh) = bboxes[i]
            # Now begin the drawing process. Pick a color, draw a rectange, and put text.
            pickedColor = [int(c) for c in colors[IDs[i]]]
            cv.rectangle(frame, (bx, by), (bx + bw, by + bh), pickedColor, 2)

            text = "{}: {:.4f}".format(labels[IDs[i]], confs[i])
            cv.putText(frame, text, (bx, by - 5), cv.FONT_HERSHEY_SIMPLEX, 0.5, pickedColor, 2)
    
# Get an image.
# "D:/YoloV4_DarkNet/darknet/data/dog.jpg"
test_frame = cv.imread("D:/download.jpg")
(h, w) = test_frame.shape[:2]
# Perform YOLO on image.
yolo_on_frame(test_frame, h, w, labels, 0.5, 0.3) 

cv.imshow("YOLO'd Image", test_frame)
cv.waitKey(0)
cv.destroyAllWindows()