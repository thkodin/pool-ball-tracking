# I need a tripod I need a tripod I need a tripod I need a trip

import os                       # for path manipulations
import sys                      # for path manipulations
import cv2 as cv                # for obvious reasons
import numpy as np              # also for obvious reasons 
import keyboard                 # for detecting keypresses
import threading                # for minimal lag on IP cameras --- couldn't manage to implement for YoloV4, scrapped since using Tiny
import multiprocessing          # for parallel processing
from imutils.video import FPS   # for metrics

# Darknet is our framework for YoloV4. Get its path.
darknetDir = 'D:/YoloV4_DarkNet/darknet'
os.add_dll_directory(darknetDir)

# Some video writing settings should we want to record stuff.
fourcc = cv.VideoWriter_fourcc(*'XVID')
recFmt = '.avi'
fps = 25

# YOLO weights, config, and name file paths...
# --------------------------------------------
namesPath = os.path.join(darknetDir, 'pbt/obj.names').replace(os.sep, '/')
cfgPath = os.path.join(darknetDir, 'pbt/yolo-obj.cfg').replace(os.sep, '/')
weightsPath = os.path.join(darknetDir, 'pbt/yolo-obj.weights').replace(os.sep, '/')
# namesPath = os.path.join(darknetDir, 'cfg/coco.names').replace(os.sep, '/')
# cfgPath = os.path.join(darknetDir, 'cfg/yolov4-tiny.cfg').replace(os.sep, '/')
# weightsPath = os.path.join(darknetDir, 'yolov4-tiny.weights').replace(os.sep, '/')

# Common parameters for all YOLO instances...
# ------------------------------------------
# Decide size of blob to be generated. Use either (320, 320), (416, 416), or (608, 608). 
# Lower means better FPS at the cost of a little accuracy...
targetSize = (416, 416)

# Initialize labels, colors, confidence and non-maxima suppression thresholds...
# Will only need one color in final implementation!
# np.random.seed(100)
# labels = np.loadtxt(namesPath, dtype = str, delimiter = ',')
# colors = np.random.randint(0, 255, size = (len(labels), 3), dtype = np.uint8)
labels = ["pool ball"]
colors = (150, 150, 255)

ConfThresh = 0.5 
NMSThresh = 0.3

# We will create a YOLO object for every camera feed. This helps keep the code organized
# and easu to understand.
class YOLO(object):
    def __init__(self):
        global namesPath; global cfgPath; global weightsPath

        # Instantiate yolo network.
        print("[INFO] Loading YOLO from", darknetDir, end = '...\n')
        self.yolo = cv.dnn.readNetFromDarknet(cfgPath, weightsPath)
        # Enable CUDA for OpenCV.
        self.yolo.setPreferableBackend(cv.dnn.DNN_BACKEND_CUDA)     
        self.yolo.setPreferableTarget(cv.dnn.DNN_TARGET_CUDA)

        # Get required output layers.
        self.layer_names = self.yolo.getLayerNames()
        self.layer_names = [self.layer_names[i - 1] for i in self.yolo.getUnconnectedOutLayers()]
    
    def detect(self, frame, fh, fw):
        # Get common parameters.
        global targetSize; global labels; global colors; global ConfThresh; global NMSThresh

        # Create a blob and perform a forward pass through the network.
        blob = cv.dnn.blobFromImage(frame, 1/255.0, (targetSize[1], targetSize[0]), swapRB = True, crop = False)
        self.yolo.setInput(blob)
        layer_outputs = self.yolo.forward(self.layer_names)  
        
        # Initialize lists to store bounding boxes, confidence scores, and class label IDs.
        bboxes = []     
        confs = []       
        IDs = []         
        
        # Get highest score per detection and extract its bounding box + label IDs...
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

        # Perform non-maxima suppression (NMS)
        bbox_idxs_to_keep = cv.dnn.NMSBoxes(bboxes, confs, ConfThresh, NMSThresh)
        if len(bbox_idxs_to_keep) > 0:
            for i in bbox_idxs_to_keep.flatten():
                # Draw bounding boxes post-NMS on the input frame.
                (bx, by, bw, bh) = bboxes[i]
                # [int(c) for c in colors[IDs[i]]]
                pickedColor = colors  
                cv.rectangle(frame, (bx, by), (bx + bw, by + bh), pickedColor, 1)
                text = "{}: {:.4f}".format(labels[IDs[i]], confs[i])
                cv.putText(frame, text, (bx, by - 5), cv.FONT_HERSHEY_SIMPLEX, 0.5, pickedColor, 1)  


# Function to display feed and actively look for keyboard presses.    
def liveFeed(cam_or_url, winName):
    global fourcc; global recFmt; global fps
    vid = cv.VideoCapture(cam_or_url)
    vid.set(cv.CAP_PROP_BUFFERSIZE, 2)              # set small buffer size
    natW = int(vid.get(cv.CAP_PROP_FRAME_WIDTH))    # get frame width
    natH = int(vid.get(cv.CAP_PROP_FRAME_HEIGHT))   # get frame height
    natFPS = int(vid.get(cv.CAP_PROP_FPS))          # get video framerate
    print("[INFO] Video Natives: {}x{} @{} FPS.".format(natW, natH, natFPS))
    
    # nextFrameDelay_sec = 1/natFPS                       # in seconds
    # nextFrameDelay_ms = int(nextFrameDelay_sec * 1000)  # in milliseconds

    # Flags related to recording options
    isPaused = False                   # for pausing recordings
    rec = None                         # if not None, then feed is being recorded to file
    n = 1                              # number of unique recording instances
    
    # Some pathing and printing QoL
    recPath = sys.path[0]                       # returns path of this script
    recName = winName + '_' + str(n) + recFmt   # to prevent overwrite of multiple recordings
    eventString = "--> " + winName + ":"        # printing QoL
    
    # Skip some frames everytime a keyboard event occurs to prevent unintentional keypresses.
    # Equivalent time (sec) = No. of frames to skip / Video framerate (0.2s at 5 frames @25 fps)
    delayEvent_keyPressed = False     # flag for events
    delayEvent_byFrames = 5           # no. of frames to delay an event by
    delayEvent_countFrame = 0         # to count up to that many frames

    # Initialize a name for the window and the YOLO network object unique to this feed.
    cv.namedWindow(winName)
    detector = YOLO()
    fps = FPS().start()

    while (vid.isOpened()):
        grabbed, frame = vid.read()
        if grabbed is False:
            print(eventString, "[ERROR]: Unable to read camera feed. Terminating...")
            break 

        # Perform YOLO object detection on frame.
        detector.detect(frame, natH, natW)
        if rec and (isPaused is False):                
            rec.write(frame)
        cv.imshow(winName, frame)
        cv.waitKey(1)
        #cv.waitKey(nextFrameDelay_ms)
        fps.update()
        #=================================================================================================
        #                                          FANCY STUFF
        #=================================================================================================
        # If an event key was pressed earlier and required frame delay was achieved...
        if delayEvent_keyPressed is False:

            # 'd' pressed while recording - discard current recording
            if keyboard.is_pressed("d") and rec:      
                delayEvent_keyPressed = True
                rec.release()
                rec = None
                delPath = os.path.join(recPath, recName)
                os.remove(delPath)
                print(eventString, "Stopped and deleted recording", recName + ".")

            # 'p' pressed while recording - pause/resume recording    
            elif keyboard.is_pressed("p") and rec:    
                delayEvent_keyPressed = True
                if isPaused is True:
                    isPaused = False
                    print(eventString, "Resumed recording", recName + ".")
                else:
                    isPaused = True
                    print(eventString, "Paused recording", recName + ".")

            # 'q' pressed - quit streaming        
            elif keyboard.is_pressed("q"):           
                # NOT CHANGING DELAY EVENT HERE BECAUSE QUITTING.
                print(eventString, "Releasing stream and saving if recording.")
                fps.stop()
                if rec:
                    rec.release()
                vid.release()
                cv.destroyWindow(winName)
                break

            # 'r' pressed - begin recording    
            elif keyboard.is_pressed("r"):           
                delayEvent_keyPressed = True
                # rec is no longer None after this.
                if rec is None:
                    rec = cv.VideoWriter(recName, fourcc, natFPS, (natW, natH))
                    print(eventString, "Started recording", recName + ".")
                else:
                    print(eventString, "Recording already started. Continuing...")

            # 's' pressed while recording - save recording        
            elif keyboard.is_pressed("s") and rec:   
                delayEvent_keyPressed = True
                # Release recording, but not the camera. 
                rec.release()
                rec = None
                n = n + 1              
                print(eventString, "Recording stopped and saved as", recName + "! Press 'r' for a new one.")
                recName = winName + '_' + str(n) + recFmt  # to not overwrite old recordings

            # Any other key pressed...   
            else:                                          
                # NOT CHANGING DELAY EVENT HERE BECAUSE NO EVENT WAS REGISTERED ON A KEY PRESS.
                continue
        #====================================================================================================
        #                                        END OF FANCY STUFF
        #====================================================================================================
            
        # If an event key was pressed earlier and required frame delay not achieved...
        else:
            delayEvent_countFrame += 1             
            # After desired no. of frames have been delayed...
            if delayEvent_countFrame % delayEvent_byFrames == 0:
                delayEvent_keyPressed = False  # reset event flag state
                delayEvent_countFrame = 0      # reset frame counter for next event 
    
    print("\n[INFO] Elasped time: {:.2f}".format(fps.elapsed()))
    print("[INFO] Average FPS: {:.2f}".format(fps.fps()))

def videoPB(path):
    name = os.path.basename(path)
    vid = cv.VideoCapture(path)
    natW = int(vid.get(cv.CAP_PROP_FRAME_WIDTH))
    natH = int(vid.get(cv.CAP_PROP_FRAME_HEIGHT))
    natFPS = int(vid.get(cv.CAP_PROP_FPS))
    print("[INFO] Video Natives: {}x{} @{} FPS.".format(natW, natH, natFPS))

    scaleW = int(0.5 * natW)
    scaleH = int(0.5 * natH)

    cv.namedWindow(name)
    detector = YOLO()
    fps = FPS().start()

    while (vid.isOpened()):
        _, frame = vid.read()
        if frame is None:
            print("[ERROR: {}] The video has ended...".format(name))
            break
        frame = cv.resize(frame, (int(scaleW), int(scaleH)))
        detector.detect(frame, scaleH, scaleW)
        cv.imshow(name, frame)
        key = cv.waitKey(natFPS)
        if key == 113:
            print("[EVENT: {}] User interrupt. Terminating...".format(name))
            break
        fps.update()
    fps.stop()
    vid.release()
    cv.destroyWindow(name)
    print("\n[INFO] Elasped time: {:.2f}".format(fps.elapsed()))
    print("[INFO] Average FPS: {:.2f}".format(fps.fps()))


if __name__ == '__main__':
    # Prompt for offline (local video) or online (live) video. 
    print("This program supports both online and offline recording and playback respectively.")
    print("---> (1) Online mode offers live feed and recording through multiple IP cameras.")
    print("---> (2) Offline mode offers playback of locally available video files.")
    pbMode = input("Choose mode (1 or 2): ")

    while pbMode not in {'1', '2'}: 
            pbMode = input("[ERROR] Please enter a valid option (1 for online or 2 for offline): ")
        
    if pbMode == '1':
        online = True
        print("ONline mode selected!\n")
    elif pbMode == '2':
        online = False
        print("OFFline mode selected!\n")

    # ONLINE MODE SELECTED.
    if online is True:
        print("============================ RECORDING CONTROLS =============================")
        print("To begin recording at any time, press 'r'. Once begun, the following applies:")
        print("\tTo pause current recording, press 'p'.")
        print("\tTo stop and save recording but continue viewing feed, press 's'.")
        print("\tTo stop and save recording (if any) as well as exit live feed, press 'q'.")
        print("\tTo discard current recording, press 'd'.")
        print("NOTE: After 's' or 'd', a new recording may be begun by pressing 'r' again.")
        print("=============================================================================")

        camT1 = multiprocessing.Process(target = liveFeed, args=("https://192.168.1.206:8080/video", 'IPCam1'))
        camT2 = multiprocessing.Process(target = liveFeed, args=('https://192.168.1.204:8081/video', 'IPCam2'))
        #camT3 = multiprocessing.Process('https://192.168.1.172:8082/video', 'IPCam3')

        camT1.start()
        camT2.start()
        #camT3.start()
        
        camT1.join()
        camT2.join()
        #camT3.join()
        print("All processes finished.")
    
    # OFFLINE MODE SELECTED.
    else:
        print("If video file is not in the notebook directory, enter the absolute path. Otherwise, just the file name would suffice.")
        print("NOTE: Enter '>' without commas to launch all the selected video files.")
        fpaths = []
        while True:
            p = input("Enter path or >: ")
            if p == '>':
                break
            fpaths.append(p)

        # Create as many empty elements as input paths in a list, and start a process
        # for each of the input video files in this list.
        vidProcesses = [[] for i in range(len(fpaths))]
        for i in range(len(vidProcesses)):
            vidProcesses[i] = multiprocessing.Process(target = videoPB, args = (fpaths[i],))
            print(vidProcesses[i])
            vidProcesses[i].start()
            
        for i in range(len(vidProcesses)):
            vidProcesses[i].join()
        
        print("[INFO] All videos have terminated. Exiting...")

# D:/YoloV4_DarkNet/Traffic2.mp4
### YOU HAVE REACHED THE END OF THE SCRIPT ###