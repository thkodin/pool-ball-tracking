import os                               # for path manipulations
import sys                              # for path manipulations
import time                             # timing stuff
import keyboard                         # for detecting keypresses
from imutils.video import FPS           # for metrics
from imutils.video import VideoStream   # later testing... supposed to be better than OpenCV's VideoCapture()

# Core functionality libs
import cv2 as cv                # for obvious reasons
import numpy as np              # also for obvious reasons 
import multiprocessing          # for parallel processing

# Custom stitching class. Initialized in imgStitcher.py, which performs image stitching on 2 images
# and returns a bunch of things (view the custom class). Can expand to 3 as well. Referenced from 
# multiple sources, but mainly OpenCV docs and pyImageSearch, as well as a blog post from towardsdatascience.
from imgStitcher import Stitcher

# DEFUNCT
#import queue        # for communication between camera and stitcher processes -- queue for threads, we use multiprocessing so we use those queues instead
#import threading    # for minimal lag on IP cameras --- couldn't manage to implement for YoloV4, scrapped since using Tiny

# Some video writing settings should we want to record stuff.
fourcc = cv.VideoWriter_fourcc(*'XVID')
recFmt = '.avi'
fps = 30

# Darknet is our framework for YoloV4, so we get its path.
darknetDir = 'D:/YoloV4_DarkNet/darknet'
os.add_dll_directory(darknetDir)

# YOLO weights, config, and name file paths
# --------------------------------------------
namesPath = os.path.join(darknetDir, 'pbt/obj.names').replace(os.sep, '/')
cfgPath = os.path.join(darknetDir, 'pbt/yolo-obj.cfg').replace(os.sep, '/')
weightsPath = os.path.join(darknetDir, 'pbt/yolo-obj.weights').replace(os.sep, '/')

# Common parameters for all YOLO instances
# ------------------------------------------
# Decide size of blob to be generated. Use either (320, 320), (416, 416), or (608, 608). 
# Lower means better FPS at the cost of a little accuracy...
targetSize = (416, 416)

# Initialize labels, colors, confidence and non-maxima suppression thresholds...
# Will only need one color in final implementation!
# To read multiple class labels conveniently...
# np.random.seed(100)
# labels = np.loadtxt(namesPath, dtype = str, delimiter = ',')
# colors = np.random.randint(0, 255, size = (len(labels), 3), dtype = np.uint8)

# For single class problems...
labels = ["pool ball"]
colors = (150, 150, 255)

# Detection and NMS thresholds
ConfThresh = 0.5 
NMSThresh = 0.3

# We will create a YOLO object for every camera feed. This helps keep the code organized
# and easy to understand. For this purpose, we create a YOLO class.
class YOLO(object):
    def __init__(self):
        global namesPath; global cfgPath; global weightsPath

        # Instantiate yolo network.
        print("[INFO-YOLO] Loading YOLO from", darknetDir, end = '...\n')
        self.yolo = cv.dnn.readNetFromDarknet(cfgPath, weightsPath)

        # Enable CUDA for OpenCV.
        self.yolo.setPreferableBackend(cv.dnn.DNN_BACKEND_CUDA)     
        self.yolo.setPreferableTarget(cv.dnn.DNN_TARGET_CUDA)

        # Get required output layers.
        self.layer_names = self.yolo.getLayerNames()
        self.layer_names = [self.layer_names[i - 1] for i in self.yolo.getUnconnectedOutLayers()]

    # Member function to detect objects in a frame.
    def detect(self, frame, fh, fw):
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
                score = det[5:]         # skip bbox info and head straight to probability scores for the labels
                ID = np.argmax(score)   # this is a consensus to assign label for multi-label problems; we leave it for expandability
                conf = score[ID]        # extract confidence for highest probable label; once again not necessary for single class problems
                
                if conf > ConfThresh:
                    bbox = det[0:4] * np.array([fw, fh, fw, fh])    # bbox has form (center_x, center_y, total_width, total_height)
                    bbox = bbox.astype("int")       
                    
                    bbox[0] = int(bbox[0] - bbox[2]/2)   # update bbox 'x' to be top left of image in horizontal direction (for drawing rect)
                    bbox[1] = int(bbox[1] - bbox[3]/2)   # update bbox 'y' to be top left of image in vertical direction (for drawing rect)
                    bboxes.append(bbox)                  
                    confs.append(float(conf))            
                    IDs.append(ID)                       

        # Perform non-maxima suppression (NMS). Note NMS also requires bbox 'x' and 'y' to be top left corner of rectangle.
        bbox_idxs_to_keep = cv.dnn.NMSBoxes(bboxes, confs, ConfThresh, NMSThresh)
        if len(bbox_idxs_to_keep) > 0:
            for i in bbox_idxs_to_keep.flatten():
                # Draw bounding boxes post-NMS on the input frame.
                (bx, by, bw, bh) = bboxes[i]
                # [int(c) for c in colors[IDs[i]]]
                pickedColor = colors  
                cv.rectangle(frame, (bx, by), (bx + bw, by + bh), pickedColor, 1)
                text = "{}: {:.2f}".format(labels[IDs[i]], confs[i])
                cv.putText(frame, text, (bx, by - 5), cv.FONT_HERSHEY_SIMPLEX, 0.5, pickedColor, 1)  


# Function to skip some frames (warm-up the camera, adjust exposure etc.)
def skipFrames(cap_obj, id_string, skip_count = 25):
    cntFrames = 1
    # Skipper loop...
    while (cap_obj.isOpened()):
        grabbed, _ = cap_obj.read()
        if grabbed is False:
            print(id_string, "[ERROR]: Unable to read a camera feed. Terminating program...")
            os._exit(0)

        # Simple counter to skip frames.
        if cntFrames <= skip_count:
            cntFrames += 1
            continue
        else:
            print(id_string, "Skipped " + str(skip_count) + " frames.")
            del(cntFrames)
            break


# For online video feed. Reads the live feed from camera, puts it in queue for stitching, and waits
# for the stitcher to return a completion event flag before repeating. Shut down any time by "q".
def liveFeed(cam_or_url, id_cam, queue, event_stitcher_ready, event_stitched):

    # Intialize capture and get some capture properties.
    vid = cv.VideoCapture(cam_or_url)
    vid.set(cv.CAP_PROP_BUFFERSIZE, 1)              # set small buffer size
    natW = int(vid.get(cv.CAP_PROP_FRAME_WIDTH))    # get frame width
    natH = int(vid.get(cv.CAP_PROP_FRAME_HEIGHT))   # get frame height
    natFPS = int(vid.get(cv.CAP_PROP_FPS))          # get video framerate
    print("[INFO-CAM] Camera Natives: {}x{} @{} FPS.".format(natW, natH, natFPS))

    eventString = "--> " + id_cam + ":"     # helps identify which camera causes any issues
    
    # Skip a second's worth of frames to allow camera adjustments and ensure both are launched properly.
    # Then, begin streaming as usual...
    #skipFrames(vid, eventString, int(natFPS))

    event_stitcher_ready.wait()     # wait for stitcher and YOLO to instantiate fully
    # Streamer loop...
    while (vid.isOpened()):
        if keyboard.is_pressed("q"):    
            print(eventString, "User interrupt registered. Terminating stream...")
            break 
        
        grabbed, frame = vid.read()     # read frame from camera
        
        # Unable to read, then...
        if grabbed is False:
            print(eventString, "[ERROR]: Unable to read camera feed. Terminating...")
            break 

        queue.put(frame)        # put the frame in the respective queue for the camera.      
        event_stitched.wait()   # wait until stitching has completed
    # Shut down video capture. This marks the end of this function.
    vid.release()


# For offline video playback and detection. Stitching is not done here as the recorded videos
# are already stitched results. If you had to, the pipeline would be pretty much the same.
def videoPB(path):
    name = os.path.basename(path)   # name the video file according to its filename                   
    vid = cv.VideoCapture(path)     # instantiate capture object

    # Get some capture properties to ensure smooth playback of each video.
    natW = int(vid.get(cv.CAP_PROP_FRAME_WIDTH))   
    natH = int(vid.get(cv.CAP_PROP_FRAME_HEIGHT))
    natFPS = int(vid.get(cv.CAP_PROP_FPS))
    print("[INFO] Video Natives: {}x{} @{} FPS.".format(natW, natH, natFPS))

    # Resize to get better output and actually fit 3 windows on the screen.
    scaleW = int(0.5 * natW)
    scaleH = int(0.5 * natH)

    # Create the named window to display video, and instantiate YOLO and FPS counter.
    cv.namedWindow(name)
    detector = YOLO()
    fps = FPS().start()

    # Begin reading the video frames.
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
    print("\n[INFO] Elapsed time: {:.2f}".format(fps.elapsed()))
    print("[INFO] Average FPS: {:.2f}".format(fps.fps()))


# For when homography has been calculated previously and some other parameters are known.
def warpAndFit(frames, H, maxRect, stitch_canvas):
    (frame_L, frame_R) = frames
    # If homography previously calculated, just do warpPerspective on right frame.
    stitched_result = cv.warpPerspective(frame_R, H, stitch_canvas)

    # Stack both frames together. This resulting image is HUGE.
    stitched_result[0 : frame_L.shape[0], 0 : frame_L.shape[1]] = frame_L

    # Some geometry, and we can mask the minimum fit rectangle as calculated by the stitcher! No blacks.
    # Change argument maxRect to minRect in function defintion and the call to it in 'stitchFeeds' func.
    #stitched_result = stitched_result[minRect[1] : minRect[1] + minRect[3], minRect[0] : minRect[0] + minRect[2]]

    # If you want the best-fit rectangle, then use this instead. Will have some black regions.
    # Change argument minRect to maxRect in function defintion and the call to it in 'stitchFeeds' func.
    stitched_result = stitched_result[maxRect[1] : maxRect[1] + maxRect[3], maxRect[0] : maxRect[0] + maxRect[2]]

    return stitched_result


def stitchFeeds(event_ready, event_stitched, id_camleft, queue_camleft, id_camright, queue_camright):
    global fourcc; global recFmt; global fps
    
    # Instantiate stitcher, motion detector (frame differencing) as well as object detector.
    stitcher = Stitcher()
    detector = YOLO()
    homography = None
    stitch_winName = "StitchedView"

    event_ready.set()       # communicate to streamers that stitcher is ready to begin reading frames
    
    # RECORDING SETTINGS
    # ==================
    isPaused = False        # for pausing recordings
    rec = None              # if not None, then feed is being recorded to file
    num_rec = 1             # number of unique recording instances
    
    # Some pathing and printing QoL
    recPath = sys.path[0]                                   # returns path of this script
    recName = stitch_winName + "_" + str(num_rec) + recFmt  # to prevent overwrite of multiple recordings
    eventString = "--> " + stitch_winName + ":"             # to identify message sender 
    
    # Prevent unintentional keypresses
    block_keyPressed = False       # flag for keypress; begin blocking
    block_forFrames = 5            # no. of frames to block interrupts by
    block_countFrame = 0           # to count how many frames have passed
    # ==================
    
    eval_fps = FPS().start()
    # Loop till user interrupt...
    while True:

        # Wait until frames available on both camera queues.
        while queue_camleft.empty() or queue_camright.empty():
            continue

        # Read the frames from both camera queues.
        frame_1 = queue_camleft.get()   # this camera must be on the left
        frame_2 = queue_camright.get()  # this camera must be on the right
        
        # Stitch the images. Image order is left first, right second. 
        if homography is None:
            # If homography not calculated before (i.e., is first frame). Can use either 1st (max fit) or 2nd 
            # (minfit) result, but don't use both at once to keep things simple. Or if you want to debug, just
            # take one as the result for the rest of the code and debug the other as a separate variable.
            stitched_result, _, maxRect, minRect, homography = stitcher.stitch([frame_1, frame_2])
            stitch_canvas = (frame_1.shape[1] + frame_2.shape[1], frame_1.shape[0] + frame_2.shape[0])
            stitched_height, stitched_width = stitched_result.shape[:2]
        else:
            #start = time.time()
            stitched_result = warpAndFit([frame_1, frame_2], homography, maxRect, stitch_canvas)
            #end = time.time()
            #print("[INFO] Stitching took {:2.4f} seconds.".format(end - start))
        
        # Perform YOLO detection on stitched image to look for pool balls.
        detector.detect(stitched_result, stitched_height, stitched_width)

        # Write to file if recording and not paused.
        if rec and (isPaused is False):                
            rec.write(stitched_result)
        
        # Show the output images.
        cv.imshow(id_camleft, frame_1)
        cv.imshow(id_camright, frame_2)
        cv.imshow(stitch_winName, stitched_result)
        cv.waitKey(1)

        # Signal stitching complete, ready camera processes for next frame.
        event_stitched.set()
        eval_fps.update()
        event_stitched.clear()  # clear stitching completion event
        
        #=================================================================================================
        #                                          FANCY STUFF
        #=================================================================================================
        # If an event key was pressed earlier and required frame delay was achieved...
        if block_keyPressed is False:

            # 'd' pressed while recording - discard current recording
            # -------------------------------------------------------
            if keyboard.is_pressed("d") and rec:      
                block_keyPressed = True
                rec.release()
                rec = None
                delPath = os.path.join(recPath, recName)
                os.remove(delPath)
                print(eventString, "Stopped and deleted recording", recName + ".")

            # 'p' pressed while recording - pause/resume recording    
            # ----------------------------------------------------
            elif keyboard.is_pressed("p") and rec:    
                block_keyPressed = True
                if isPaused is True:
                    isPaused = False
                    print(eventString, "Resumed recording", recName + ".")
                else:
                    isPaused = True
                    print(eventString, "Paused recording", recName + ".")

            # 'q' pressed - quit streaming       
            # ---------------------------- 
            elif keyboard.is_pressed("q"):           
                # NOT CHANGING DELAY EVENT HERE BECAUSE QUITTING.
                print(eventString, "Releasing stream and saving if recording.")
                break

            # 'r' pressed - begin recording
            # -----------------------------    
            elif keyboard.is_pressed("r"):           
                block_keyPressed = True
                # rec is no longer None after this.
                if rec is None:
                    rec = cv.VideoWriter(recName, fourcc, fps, (int(stitched_width), int(stitched_height)))
                    print(eventString, "Started recording", recName + ".")
                else:
                    print(eventString, "Recording already started. Continuing...")

            # 's' pressed while recording - save recording   
            # --------------------------------------------     
            elif keyboard.is_pressed("s") and rec:   
                block_keyPressed = True
                # Release recording, but not the camera. 
                rec.release()
                # Prep for a new recording beforehand.
                rec = None
                num_rec += 1              
                print(eventString, "Recording stopped and saved as", recName + "! Press 'r' for a new one.")
                recName = stitch_winName + "_" + str(num_rec) + recFmt  # to not overwrite old recordings

            # Any other key pressed...  
            # ------------------------ 
            else:                                          
                # NOT CHANGING DELAY EVENT HERE BECAUSE NO EVENT WAS REGISTERED ON A KEY PRESS.
                continue

        # If an event key was pressed earlier and required frame delay not achieved...
        else:
            block_countFrame += 1             
            # After desired no. of frames have been delayed...
            if block_countFrame % block_forFrames == 0:
                block_keyPressed = False  # reset event flag state
                block_countFrame = 0      # reset frame counter for next event 
        #=================================================================================================
        #                                       END OF FANCY STUFF
        #=================================================================================================

    # Release recorder object if recording, destroy all windows, and print out the FPS.
    eval_fps.stop()
    if rec:
        rec.release()

    cv.destroyWindow(stitch_winName)
    cv.destroyWindow(id_camleft)
    cv.destroyWindow(id_camright)

    print("\n[INFO] Elapsed time: {:.2f}".format(eval_fps.elapsed()))
    print("[INFO] Average FPS: {:.2f}".format(eval_fps.fps()))
    

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
        print("=============================================================================\n")
        
        # Define names for the camera windows. These are used later to identify which queue to fill.
        name_cam1 = 'IPCam1'
        name_cam2 = 'IPCam2'

        # Initialize queues and events for syncing frames between cameras and communication with stitcher process.
        queue_cam1 = multiprocessing.Queue(maxsize = 1)
        queue_cam2 = multiprocessing.Queue(maxsize = 1)
        stitcher_ready = multiprocessing.Event()
        stitched = multiprocessing.Event()

        # MAKE SURE camP1 gets a left view of the scene, and camP2 gets the right view.
        camP1 = multiprocessing.Process(target = liveFeed, args=("https://192.168.1.206:8080/video", 
                                        name_cam1, queue_cam1, stitcher_ready, stitched))
        camP2 = multiprocessing.Process(target = liveFeed, args=('https://192.168.1.205:8081/video', 
                                        name_cam2, queue_cam2, stitcher_ready, stitched))

        stchP = multiprocessing.Process(target = stitchFeeds, args=(stitcher_ready, stitched, 
                                        name_cam1, queue_cam1, name_cam2, queue_cam2))

        # Start the processes.
        camP1.start()
        camP2.start()
        stchP.start()
        
        # Wait for them to end.
        stchP.join()
        camP1.join()
        camP2.join()

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
            else:
                fpaths.append(p) if os.path.exists(p) else print("[ERROR] Invalid path received. Ensure path/name is correct.")

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

### YOU HAVE REACHED THE END OF THE SCRIPT ###