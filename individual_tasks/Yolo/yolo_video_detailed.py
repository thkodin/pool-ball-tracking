########################## EXPLANATORY CODE BLOCK FOR YOLO ON A VIDEO ################################

import os           
import sys          
import cv2 as cv    
import numpy as np
from imutils.video import FPS

cudaDir = 'C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.5/bin'
darknetDir = 'D:/YoloV4_DarkNet/darknet'

os.add_dll_directory(cudaDir)        # dependency of darknet 
os.add_dll_directory(darknetDir)     # darknet itself

scriptPath = sys.path[0]

# Define the required file paths.
print("==============================================================")
namesPath = os.path.join(darknetDir, 'cfg/coco.names').replace(os.sep, '/')
print("(Names)\t\t", namesPath)
cfgPath = os.path.join(darknetDir, 'yolov4.cfg').replace(os.sep, '/')
print("(Configs)\t", cfgPath)
weightsPath = os.path.join(darknetDir, 'yolov4.weights').replace(os.sep, '/')
print("(Weights)\t", weightsPath)
print("==============================================================")

# Instantiate yolo network. Each camera needs a unique instantiation.
print("[INFO] Loading YOLO from", darknetDir, end = '...\n')
yolo = cv.dnn.readNetFromDarknet(cfgPath, weightsPath)
# Enable CUDA for OpenCV.
yolo.setPreferableBackend(cv.dnn.DNN_BACKEND_CUDA)     
yolo.setPreferableTarget(cv.dnn.DNN_TARGET_CUDA)

# Get required output layers.
layer_names = yolo.getLayerNames()
layer_names = [layer_names[i - 1] for i in yolo.getUnconnectedOutLayers()]


### GLOBALIZE TARGET SIZE, LABELS, COLOR, MINCONF, NMSTHRESH ###
### ======================================================== ###
# Use either (320, 320), (416, 416), or (608, 608). Lower means better FPS at the cost
# of a little accuracy.
targetSize = (416, 416)

# Initialize labels and colors.
np.random.seed(100)
labels = np.loadtxt(namesPath, dtype = str, delimiter = ',')
colors = np.random.randint(0, 255, size = (len(labels), 3), dtype = np.uint8)
ConfThresh = 0.5 
NMSThresh = 0.3
### ========================================================= ###

# Perfrom yolo on a frame using a particular instantiation of the network.
def yolo_on_frame(net, frame, fw, fh):
    # Get global settings. This is to synchronize YOLO settings across all cameras.
    global targetSize; global labels; global colors; global ConfThresh; global NMSThresh 
    # Perform a forward pass with the input's blob.
    blob = cv.dnn.blobFromImage(frame, 1/255.0, (targetSize[1], targetSize[0]), swapRB = True, crop = False)
    net.setInput(blob)
    layer_outputs = net.forward(layer_names)  
    
    # Initialize lists to store bounding boxes, confidence scores, and class label IDs.
    bboxes = []     
    confs = []       
    IDs = []         
    
    # Get highest scores and their bounding boxes + label IDs...
    for output in layer_outputs:
        for det in output:
            score = det[5:]
            ID = np.argmax(score)          
            conf = score[ID]              
            
            if conf > ConfThresh:
                bbox = det[0:4] * np.array([fw, fh, fw, fh])
                bbox = bbox.astype("int")       
                
                bbox[0] = int(bbox[0] - bbox[2]/2)   
                bbox[1] = int(bbox[1] - bbox[3]/2)                                 
                bboxes.append(bbox)                  
                confs.append(float(conf))            
                IDs.append(ID)                       

    # Perform non-maxim suppression
    bbox_idxs_to_keep = cv.dnn.NMSBoxes(bboxes, confs, ConfThresh, NMSThresh)
    if len(bbox_idxs_to_keep) > 0:
        for i in bbox_idxs_to_keep.flatten():
            # Draw bounding boxes post-NMS on the input frame.
            (bx, by, bw, bh) = bboxes[i]
            pickedColor = [int(c) for c in colors[IDs[i]]]
            cv.rectangle(frame, (bx, by), (bx + bw, by + bh), pickedColor, 2)
            text = "{}: {:.4f}".format(labels[IDs[i]], confs[i])
            cv.putText(frame, text, (bx, by - 5), cv.FONT_HERSHEY_SIMPLEX, 0.5, pickedColor, 2)       


# Initialize camera as capture object.
# D:/YoloV4_DarkNet/darknet/Traffic1.mp4
cap = cv.VideoCapture("https://192.168.1.199:8080/video")
#cap.set(cv.CAP_PROP_BUFFERSIZE, 2)
natW = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
natH = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
natFPS = int(cap.get(cv.CAP_PROP_FPS))
print("[INFO] Camera Natives: {}x{} @{} FPS.".format(natW, natH, natFPS))


fps = FPS().start()             # launch fps measurements
while True:                     # begin reading
    grabbed, frame = cap.read()

    # Exit if frame not read. 
    if grabbed is False:
        print("[ERROR] Unable to read frame. Terminating...")
        break
    
    # Apply yolo on frame. We give the natW and natH to save computation 
    # of the width and height on every frame.
    yolo_on_frame(yolo, frame, natW, natH) 

    # Show frame with detections.
    cv.imshow("YOLO'd Video", frame)

    # Exit on user interrupt.
    key = cv.waitKey(1)     
    if key == ord('q'):
        print("[EVENT] User interrupt. Terminating stream...")
        break

    # Update fps.
    fps.update()            

fps.stop()
cap.release()
cv.destroyAllWindows()
print("\n[INFO] Elasped time: {:.2f}".format(fps.elapsed()))
print("[INFO] Average FPS: {:.2f}".format(fps.fps()))
