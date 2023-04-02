# Every part of this program has a very heavily commented script included in the 'individual_tasks' folder.
# We implemented each function separately before integrating into our script, and in the separate implementation
# we commented everything we learnt regarding the specific functionality we were introducing in the context of
# coding. So the comments will be a bit light here. For more details, you may view the aforementioned scripts.

# Pathing, QoL, user interrupt libs.
import os                               # for path manipulations
import sys                              # for path manipulations
import time                             # timing stuff
import keyboard                         # for detecting keypresses
from imutils.video import FPS           # for metrics

# Core functionality libs.
import cv2 as cv                        # for obvious reasons
import numpy as np                      # also for obvious reasons 
import multiprocessing                  # for parallel processing

# Custom stitching class. Initialized in imgStitcher.py, which performs image stitching on 2 images
# and returns a bunch of things (view the custom class). Can expand to 3 as well. Referenced from 
# multiple sources, but mainly OpenCV docs and pyImageSearch, as well as a blog post from towardsdatascience.
from imgStitcher import Stitcher
import queue                        # used in maintaining heatmap history

# Unused now...
#import threading    # for minimal lag on IP cameras with threads --- couldn't manage to implement for YoloV4, scrapped since using Tiny
#from imutils.video import VideoStream   # supposed to be better than OpenCV's VideoCapture()

# np.random.seed(100)

# Some video writing settings should we want to record stuff.
fourcc = cv.VideoWriter_fourcc(*'XVID')
recFmt = '.avi'
fps = 30

# Darknet is our framework for YoloV4 and all our configs are there, so we get its path.
darknetDir = 'D:/YoloV4_DarkNet/darknet'
os.add_dll_directory(darknetDir)            # only necessary if using YOLO on command line interface; useful for training

# YOLO weights, config, and name file paths
# -----------------------------------------
namesPath = os.path.join(darknetDir, 'pbt/obj.names').replace(os.sep, '/')
cfgPath = os.path.join(darknetDir, 'pbt/yolo-obj.cfg').replace(os.sep, '/')
weightsPath = os.path.join(darknetDir, 'pbt/yolo-obj.weights').replace(os.sep, '/')

# Common parameters for all YOLO instances
# ----------------------------------------
# Decide size of blob to be generated. Use either (320, 320), (416, 416), or (608, 608), or 
# any multiple of 32 to allow YOLO to work with its 'grid' formula. Lower means better FPS
# but worse accuracy. Keep in mind we had no control samples in our training data, so this
# particular network will call ANY CIRCULAR/SPHERICAL SHAPE A POOL BALL.
targetSize = (416, 416)

# Detection and NMS thresholds
ConfThresh = 0.5 
NMSThresh = 0.3

# YOLO class to keep things concise. You may event put this in another script and import it from
# there, just like the stitcher class to keep this script shorter.
class YOLO(object):
    def __init__(self, weightsPath, cfgPath, namesPath, ConfThresh=0.5, NMSThresh=0.3):
        self.ConfThresh = ConfThresh
        self.NMSThresh = NMSThresh

        # Initialize labels, colors, confidence and non-maxima suppression thresholds...
        # To read multiple class labels conveniently...
        # Note: Haven't tested with '\r\n'; works with ',' but only for multi-class case.
        # self.labels = np.loadtxt(namesPath, dtype = str, delimiter = '\r\n').tolist()

        # For single class problems...
        self.labels = ['pool ball']

        if len(self.labels) == 1:
            self.colors = [150, 150, 255]
        else:
            self.colors = np.random.randint(0, 255, size = (len(self.labels), 3), dtype = np.uint8).tolist()

        # Instantiate YOLO network.
        print('[INFO-YOLO] Loading YOLO from', darknetDir, end = '...\n')
        self.yolo = cv.dnn.readNetFromDarknet(cfgPath, weightsPath)

        # Enable CUDA for OpenCV.
        self.yolo.setPreferableBackend(cv.dnn.DNN_BACKEND_CUDA)     
        self.yolo.setPreferableTarget(cv.dnn.DNN_TARGET_CUDA)

        # Get required output layers.
        self.layer_names = self.yolo.getLayerNames()
        self.layer_names = [self.layer_names[i - 1] for i in self.yolo.getUnconnectedOutLayers()]

    # Member function to detect objects in a frame.
    def detect(self, frame, fh, fw, targetSize):

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
                
                if conf > self.ConfThresh:
                    bbox = det[0:4] * np.array([fw, fh, fw, fh])    # bbox has form (center_x, center_y, total_width, total_height)
                    bbox = bbox.astype('int')       
                    
                    bbox[0] = int(bbox[0] - bbox[2]/2)   # update bbox 'x' to be top left of image in horizontal direction (for drawing rect)
                    bbox[1] = int(bbox[1] - bbox[3]/2)   # update bbox 'y' to be top left of image in vertical direction (for drawing rect)
                    bboxes.append(bbox)                  
                    confs.append(float(conf))            
                    IDs.append(ID)                       

        # Perform non-maxima suppression (NMS). Note NMS also requires bbox 'x' and 'y' to be top left corner of rectangle.
        bbox_idxs_to_keep = cv.dnn.NMSBoxes(bboxes, confs, self.ConfThresh, self.NMSThresh)
        if len(bbox_idxs_to_keep) > 0:
            for i in bbox_idxs_to_keep.flatten():
                # Draw bounding boxes post-NMS on the input frame.
                (bx, by, bw, bh) = bboxes[i]

                if len(self.labels==1):
                    pickedColor = self.colors
                else:
                    pickedColor = [int(c) for c in self.colors[IDs[i]]]  # cv2.rectangle() requires int, not numpy.uint8
  
                cv.rectangle(frame, (bx, by), (bx + bw, by + bh), pickedColor, 1)
                text = '{}: {:.2f}'.format(self.labels[IDs[i]], confs[i])
                cv.putText(frame, text, (bx, by - 5), cv.FONT_HERSHEY_SIMPLEX, 0.5, pickedColor, 1)  


# Function to skip some frames (warm-up the camera, adjust exposure etc.). Not always needed.
def skipFrames(cap_obj, id_string, skip_count = 25):
    cntFrames = 1
    # Skipper loop...
    while (cap_obj.isOpened()):
        grabbed, _ = cap_obj.read()
        if grabbed is False:
            print(id_string, '[ERROR]: Unable to read a camera feed. Terminating program...')
            os._exit(0)

        # Simple counter to skip frames.
        if cntFrames <= skip_count:
            cntFrames += 1
            continue
        else:
            print(id_string, 'Skipped ' + str(skip_count) + ' frames.')
            del(cntFrames)
            break


# For online video feed. Reads the live feed from camera, puts it in queue for stitching, and waits
# for the stitcher to return a completion event flag before repeating. Shut down any time by 'q'.
def liveFeed(cam_or_url, id_cam, queue, event_stitcher_ready, event_stitched):

    # Intialize capture and get some capture properties.
    vid = cv.VideoCapture(cam_or_url)
    vid.set(cv.CAP_PROP_BUFFERSIZE, 1)              # set small buffer size
    natW = int(vid.get(cv.CAP_PROP_FRAME_WIDTH))    # get frame width
    natH = int(vid.get(cv.CAP_PROP_FRAME_HEIGHT))   # get frame height
    natFPS = int(vid.get(cv.CAP_PROP_FPS))          # get video framerate
    print('[INFO-CAM] Camera Natives: {}x{} @{} FPS.'.format(natW, natH, natFPS))

    eventString = '--> ' + id_cam + ':'     # helps identify which camera causes any issues
    
    # Skip a second's worth of frames to allow camera adjustments and ensure both are launched properly.
    # Then, begin streaming as usual... not a necessary feature but left in regardless.
    #skipFrames(vid, eventString, int(natFPS))

    event_stitcher_ready.wait()     # wait for stitcher and YOLO to instantiate fully
    # Streamer loop...
    while (vid.isOpened()):
        grabbed, frame = vid.read()     # read frame from camera
        queue.put(frame)                # put the frame in the respective queue for the camera.      
        event_stitched.wait()           # wait until stitching has completed

        # Unable to read, then...
        if grabbed is False:
            print(eventString, '[ERROR]: Unable to read camera feed. Terminating...')
            break 
        if keyboard.is_pressed('q'):    
            print(eventString, 'User interrupt registered. Terminating stream...')
            break 

    # Shut down video capture. This marks the end of this function.
    vid.release()


# For offline video playback and detection. Stitching is not done here as the recorded videos
# are already stitched results. If you had to stitch, the pipeline would be pretty much the same.
def videoPB(path):
    name = os.path.basename(path)   # name the video file according to its filename                   
    vid = cv.VideoCapture(path)     # instantiate capture object

    # Get some capture properties to ensure smooth playback of each video.
    natW = int(vid.get(cv.CAP_PROP_FRAME_WIDTH))   
    natH = int(vid.get(cv.CAP_PROP_FRAME_HEIGHT))
    natFPS = int(vid.get(cv.CAP_PROP_FPS))
    print('[INFO] Video Natives: {}x{} @{} FPS.'.format(natW, natH, natFPS))

    # Resize to get better output and actually fit 3 windows on the screen.
    scaleW = int(0.5 * natW)
    scaleH = int(0.5 * natH)

    # Create the named window to display video, and instantiate YOLO and FPS counter.
    cv.namedWindow(name)
    detector = YOLO(weightsPath=weightsPath, 
                    cfgPath=cfgPath, 
                    namesPath=namesPath, 
                    ConfThresh=ConfThresh, 
                    NMSThresh=NMSThresh)
    
    fps = FPS().start()
    # Begin reading the video frames.
    while (vid.isOpened()):
        _, frame = vid.read()

        if frame is None:
            print('[ERROR: {}] The video has ended...'.format(name))
            break

        frame = cv.resize(frame, (int(scaleW), int(scaleH)))
        detector.detect(frame, scaleH, scaleW, targetSize=targetSize)
        cv.imshow(name, frame)

        key = cv.waitKey(natFPS)    # to ensure proper FPS on video playback, otherwise it speeds up or slows down
        if key == 113:
            print('[EVENT: {}] User interrupt. Terminating...'.format(name))
            break
        fps.update()

    fps.stop()
    vid.release()
    cv.destroyWindow(name)
    print('\n[INFO] Elapsed time: {:.2f}'.format(fps.elapsed()))
    print('[INFO] Average FPS: {:.2f}'.format(fps.fps()))


# For when homography has been calculated previously and some other parameters are known. Note that this
# particular function is for stitching, since the post-stitch process needs a little bit of refinement.
def warpAndFit(frames, H, maxRect, stitch_canvas):
    (frame_L, frame_R) = frames
    # If homography previously calculated, just do warpPerspective on right frame.
    # Note that the image size for warping MUST be (width, height), not (height, width)!
    stitched_result = cv.warpPerspective(frame_R, H, stitch_canvas)

    # Stack both frames together. This resulting image is HUGE.
    stitched_result[0 : frame_L.shape[0], 0 : frame_L.shape[1]] = frame_L

    # Some geometry, and we can mask the minimum fit rectangle as calculated by the stitcher! No blacks.
    # Change argument maxRect to minRect in function defintion and the call to it in 'stitchFeeds' func.
    #stitched_result = stitched_result[minRect[1] : minRect[1] + minRect[3], minRect[0] : minRect[0] + minRect[2]]

    # If you want the best-fit rectangle, use this instead. Will have some black regions, but is generally the safest.
    # Change argument minRect to maxRect in function defintion and the call to it in 'stitchFeeds' func.
    stitched_result = stitched_result[maxRect[1] : maxRect[1] + maxRect[3], maxRect[0] : maxRect[0] + maxRect[2]]

    return stitched_result


# Please note that all the arguments are for multiprocess resource sharing and nothing else.
# Defining these frameworks as global objects doesn't work (we tried), so you have to include any shared queues
# and events in all the arguments for these multiprocessing function targets.

def stitchFeeds(event_ready, event_stitched, id_camleft, queue_camleft, id_camright, queue_camright):
    # Import video recording settings.
    global fourcc; global recFmt; global fps    

    '''
    Mouse event callback function to mark 4 points on the planar object. These points are then used to compute the bird's eye
    homography for warping. This is done on the STITCHED image, and ONLY ONCE with the assumption that the cameras do not move.
    Note: Mouse Callbacks expect 5 arguments in the callback function description and throw errors otherwise. DO NOT REMOVE.

    MARK ORDER IS SUPER IMPORTANT! 
    --> 1st Point: TOP LEFT
    --> 2nd Point: TOP RIGHT
    --> 3rd Point: BOTTOM LEFT
    --> 4th Point: BOTTOM RIGHT
    Press any key once all points are marked to proceed.

    Also note that cv2.getPerspectiveTransform requires two main things: the point pixel coordinates in the original image, and
    the point coordinates in the final output image. The first set is what this function helps identify. The second set is then
    inferred from this given that we want the rectangle defined by these points as our final image view (bird's eye). BOTH SETS
    MUST BE FLOAT-32 VALUES, OTHERWISE NOTHING WORKS!

    You may also note that getPerspectiveTransform and findHomography (used in stitching) do the same thing (i.e. compute a
    homography), but the main difference is that gPT is meant to work with just 4 points very quickly, whereas findHomography
    is optimized for overconstrained systems with more than 4 points of interest, as in finding correspondences. 
    '''

    def markPoints(event, x, y, flags, param):                  
        if event == cv.EVENT_LBUTTONDOWN:                           # on left-click (downpress only)
            cv.circle(first_frame, (x, y), 5, (150, 150, 255), -1)  # helps visualize marks as you go

            if len(points) < 4:         # technically you need arrays, but we just use np.asanyarray(list)
                points.append([x, y])
            elif len(points) == 4:
                print('[INFO-BIRD] Already marked 4 points. Press any key to exit...')

            cv.imshow(mark_winName, first_frame)  # update the display image with marked points

    # ---------------
    # INITIALIZATIONS
    # ===============
    stitcher = Stitcher()                               # custom stitcher object
    detector = YOLO(weightsPath=weightsPath, 
                    cfgPath=cfgPath, 
                    namesPath=namesPath, 
                    ConfThresh=ConfThresh, 
                    NMSThresh=NMSThresh)
    
    homography_stitch = None                            # store homography for stitch warp
    homography_top = None                               # store homography for bird's eye warp
    stitch_winName = 'stitch_view'                      # to display stitched results (if needed)
    top_winName = 'top_view'                            # to display top-view generation with detected objects 
    mark_winName = 'Mark 4 Points (TL, TR, BL, BR)'     # display window for marking top-view points

    # Create the background subtractor object.
    # varThreshold will need a bit of tweaking now and then for different setups. For better robustness, use the morphological
    # expansions to filter noise rather than the distance threshold.
    # defaults: history = 500, varThreshold = 16, detectShadows = True
    bgSub = cv.createBackgroundSubtractorMOG2(varThreshold = 30, detectShadows = False)

    points = []                                                     # store the 4 marked points for top-view homography
    hmap_history_in_secs = 10                                       # history of object motion heatmap to maintain (in seconds)
    krnl = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))       # for filtering noise (shape of kernel, kernel size)
    queue_hmap = queue.Queue(maxsize = hmap_history_in_secs * fps)  # actual history of heatmap kept in this queue structure

    event_ready.set()   # communicate to streamers that stitcher is ready to begin reading frames
    # ------------------
    # RECORDING SETTINGS
    # ==================
    isPaused = False    # for pausing recordings
    rec = None          # if not None, then feed is being recorded to file
    rec_hmp = None      # to record heatmap
    num_rec = 1         # number of unique recording instances to avoid overwriting previous ones in the same session
    
    # Some pathing and printing QoL
    recPath = sys.path[0]                                        # returns path of this script
    recName = top_winName + '_' + str(num_rec) + recFmt          # to prevent overwrite of multiple recordings
    rec_hmpName = top_winName + '_hmp_' + str(num_rec) + recFmt  # to prevent overwrite of multiple recordings
    eventString = '--> ' + top_winName + ':'                     # to identify message sender 
    
    # Unintentional keypresses
    block_keyPressed = False    # set whenever a key is pressed, cleared after a number of frames passed
    block_forFrames = 5         # no. of frames to block any further keypresses by
    block_countFrame = 0        # to count how many frames have passed
    # ------------------
    # ACTUAL MAGIC
    # ==================
    eval_fps = FPS().start()    # initialize FPS evaluation object

    # Loop till user interrupt...
    while True:

        # Wait until frames available on both camera queues.
        while queue_camleft.empty() or queue_camright.empty():
            continue

        frame_left = queue_camleft.get()    # this camera MUST be on the LEFT
        frame_right = queue_camright.get()  # this camera MUST be on the RIGHT

        # If homographies not calculated before (i.e., is first frame). Can use either 1st (maxfit) or 2nd (minfit) result of
        # stitcher. Minfit is basically a hack, so if you get weird results just use maxfit which is guaranteed to be a safe 
        # result. Further, the user must mark points on the stitched result to calculate the bird's eye homography as well.

        if homography_stitch is None or homography_top is None:

            # IMAGE STITCHING (HOMOGRAPHY)
            # ============================
            # Stitch the images. Image order is left first, right second. This is how our particular class is implemented,
            # so if you want to switch the order be sure to make the necessary changes! The left camera image is NOT transformed
            # while the right camera image is warped to fit to the right side of the left camera image in the final canvas.

            # To use minfit results, set 2nd return to stitched_result and first to _. Do the reverse for maxfit.
            stitched_result, _, maxRect, minRect, homography_stitch = stitcher.stitch([frame_left, frame_right])

            # (wdith, height) is required for warping so we extract the canvas in that order as well.
            stitch_canvas = (frame_left.shape[1] + frame_right.shape[1], frame_left.shape[0] + frame_right.shape[0])
            
            # This is used for display, so we use the conventional (height, width) here.
            stitched_height, stitched_width = stitched_result.shape[:2]

            # BIRD'S EYE VIEW (HOMOGRAPHY)
            # ============================
            # We assume the first frame to be a good representation of our background model. We will use high-level CV2 API here
            # to implement mixture of gaussian (MoG), nothing too fancy. Of course, you can change the history of Gaussians
            # maintained (longer history means your object will be more likely considered as background if it doesn't move much),
            # the amount of movement to be considered as non-static (varThreshold), and shadow detection etc. We will keep shadow
            # detection off to save FPS, and keep the history shorter as well since pool balls do remain static for a while.

            first_frame = stitched_result.copy()            # copy stitched result for marking
            cv.imshow(mark_winName, first_frame)            # show the frame to mark points in
            cv.setMouseCallback(mark_winName, markPoints)   # set the callback to this window on to function markPoints
            cv.waitKey(0)                                   # for manual user exit after marking 4 points
            cv.destroyWindow(mark_winName)                  # destroy the marking window ONLY

            top_height, top_width =  points[2][1] - points[0][1], points[1][0] - points[0][0]    # height, width (must be +ve)

            # Notice how both point sets are Float-32 numpy ARRAYS, not lists or tuples or anything else. Notice (w, h) order.
            points = np.float32(np.asanyarray(points))
            points_top = np.array(([0, 0], [top_width, 0], [0, top_height], [top_width, top_height]), np.float32)

            # Calculate homography for top-view from the two point sets above and initialize the image to store the motion
            # as binary. We'll apply a colormap to it later!
            homography_top = cv.getPerspectiveTransform(points, points_top)
            accum_image = np.zeros((top_height, top_width), np.uint8)

        # If not first frame, use previously calculated homographies. Yeah, it's big brain time!
        else:
            #start = time.time()
            stitched_result = warpAndFit([frame_left, frame_right], homography_stitch, maxRect, stitch_canvas)
            #end = time.time()
            #print('[INFO] Stitching took {:2.4f} seconds.'.format(end - start))

        # To keep track of just the pool balls, it might be better to just maintain a history of every detected pool
        # ball's pixel movement points rather than generate a heatmap. It would just require the bboxes we get from
        # YOLO, and is probably more useful. Regardless, we'll go with motion heatmaps here.
        #start = time.time()
        frame_top = cv.warpPerspective(stitched_result, homography_top, (top_width, top_height))

        # Remove oldest motion effect from heatmap (works because we're dealing with binary images)
        if queue_hmap.full():
            accum_image = cv.subtract(accum_image, queue_hmap.get())
        
        bgSubbed = bgSub.apply(frame_top)
        
        # Use if MOG varThreshold doesn't cut it. MorphEx is just a fancy name for erosion-dilation series to help 
        # remove noise (small white spots get eroded away, and actual objects are dilated back to original shape).
        bgSubbed_lessNoise = cv.morphologyEx(bgSubbed, cv.MORPH_OPEN, krnl)     

        # Create a binary threshold to filter out noise from the model.
        _, thresh = cv.threshold(bgSubbed, 250, 100, cv.THRESH_BINARY)

        # Accumulate the heatmap image every frame.
        accum_image = cv.add(accum_image, thresh)
    
        # Add the filter threshold to history queue.
        queue_hmap.put(thresh)

        # Provide a colormap to the masked image. We do this before YOLO because our implementation of YOLO draws 
        # the bounding boxes on the image directly, which then are incorrectly detected as moving objects in the heatmap.
        heatmapped = cv.applyColorMap(accum_image, cv.COLORMAP_SUMMER)
        
        # Perform YOLO detection on stitched image (non-heatmap) to look for pool balls. Draws bboxes around detections.
        # You may modify the class function to return these bboxes for implementing KCF or other trackers if FPS are
        # a problem, or even verify if a pool ball was pocketed (draw a rectangle around pool table pockets, if any ball's
        # center enters this it counts as a pocket). Might implement in the future.
        detector.detect(frame_top, top_height, top_width, targetSize=targetSize)
        
        # Write to file if recording and not paused.
        if rec and rec_hmp and not isPaused:                
            rec.write(frame_top)
            rec_hmp.write(heatmapped)
        
        # Show the output images.
        #cv.imshow(id_camleft, frame_left)           # unnecessary
        #cv.imshow(id_camright, frame_right)         # unnecessary
        #cv.imshow(stitch_winName, stitched_result)  # unnecessary

        cv.imshow(top_winName, frame_top)           # necessary
        cv.imshow('Motion Heatmap', heatmapped)     # necessary
        cv.waitKey(1)                               
        
        event_stitched.set()    # signal stitching complete, ready camera processes for next frame.
        eval_fps.update()       # update fps
        event_stitched.clear()  # clear stitching completion event
        

        #end = time.time()
        #print('[INFO] Stitching took {:2.4f} seconds.'.format(end - start))
        # Ideally, you'd do the below (interrupt checking) in a separate thread, but whatever.
        #=================================================================================================
        #                                    FANCY STUFF (USER OPTIONS)
        #=================================================================================================
        # If no key pressed, or if a key was pressed earlier and required frame delay was achieved...
        if block_keyPressed is False:

            # 'd' pressed while recording - discard current recording
            # -------------------------------------------------------
            if keyboard.is_pressed('d') and rec:      
                block_keyPressed = True
                rec.release(); rec_hmp.release()
                rec = None; rec_hmp = None
                delPath = os.path.join(recPath, recName); os.remove(delPath)
                delPath = os.path.join(recPath, rec_hmpName); os.remove(delPath)
                print(eventString, 'Stopped and deleted recording', recName + '.')

            # 'p' pressed while recording - pause/resume recording    
            # ----------------------------------------------------
            elif keyboard.is_pressed('p') and rec:    
                block_keyPressed = True
                if isPaused is True:
                    isPaused = False
                    print(eventString, 'Resumed recording', recName + '...')
                else:
                    isPaused = True
                    print(eventString, 'Paused recording', recName + '...')

            # 'q' pressed - quit streaming       
            # ---------------------------- 
            elif keyboard.is_pressed('q'):           
                # NOT CHANGING DELAY EVENT HERE BECAUSE QUITTING.
                print(eventString, 'Releasing stream and saving if recording.')
                break

            # 'r' pressed - begin recording
            # -----------------------------    
            elif keyboard.is_pressed('r'):           
                block_keyPressed = True
                # rec is no longer None after this.
                if None in {rec, rec_hmp}:
                    rec = cv.VideoWriter(recName, fourcc, fps, (int(top_width), int(top_height)))
                    rec_hmp = cv.VideoWriter(rec_hmpName, fourcc, fps, (int(top_width), int(top_height)))
                    print(eventString, 'Started recording', recName + '.')
                else:
                    print(eventString, 'Recording already started. Continuing...')

            # 's' pressed while recording - save recording   
            # --------------------------------------------     
            elif keyboard.is_pressed('s') and rec:   
                block_keyPressed = True
                # Release recording, but not the camera. 
                rec.release(); rec_hmp.release()
                # Prep for a new recording beforehand.
                rec = None; rec_hmp = None
                num_rec += 1              
                print(eventString, 'Recording stopped and saved as', recName + '! Press 'r' for a new one.')
                recName = top_winName + '_' + str(num_rec) + recFmt      # to not overwrite old recordings
                recName = top_winName + '_hmp_' + str(num_rec) + recFmt  # same for heatmap recordings

            # Any other key pressed 
            # ---------------------
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
    if rec and rec_hmp:
        rec.release()
        rec_hmp.release()

    cv.destroyWindow(top_winName)
    cv.destroyWindow('Motion Heatmap')
    #cv.destroyWindow(stitch_winName)
    #cv.destroyWindow(id_camleft)
    #cv.destroyWindow(id_camright)

    print('\n[INFO] Elapsed time: {:.2f}'.format(eval_fps.elapsed()))
    print('[INFO] Average FPS: {:.2f}'.format(eval_fps.fps()))
    

if __name__ == '__main__':
    # Prompt for offline (local video) or online (live) video. 
    print('This program supports both online and offline recording and playback respectively.')
    print('---> (1) Online mode offers live feed and recording through multiple IP cameras.')
    print('---> (2) Offline mode offers playback of locally available video files.')
    pbMode = input('Choose mode (1 or 2): ')

    while pbMode not in {'1', '2'}: 
            pbMode = input('[ERROR] Please enter a valid option (1 for online or 2 for offline): ')
        
    if pbMode == '1':
        online = True
        print('ONline mode selected!\n')
    elif pbMode == '2':
        online = False
        print('OFFline mode selected!\n')

    # ONLINE MODE SELECTED.
    if online is True:
        print('============================ RECORDING CONTROLS =============================')
        print('To begin recording at any time, press "r". Once begun, the following applies:')
        print('\tTo pause current recording, press "p".')
        print('\tTo stop and save recording but continue viewing feed, press "s".')
        print('\tTo stop and save recording (if any) as well as exit live feed, press "q".')
        print('\tTo discard current recording, press "d".')
        print('NOTE: After "s" or "d", a new recording may be begun by pressing "r" again.')
        print('=============================================================================\n')
        
        # Define names for the camera windows. These are used later to identify which queue to fill.
        name_cam1 = 'IPCam1'
        name_cam2 = 'IPCam2'

        # Initialize queues and events for syncing frames between cameras and communication with stitcher process.
        queue_cam1 = multiprocessing.Queue(maxsize = 1)
        queue_cam2 = multiprocessing.Queue(maxsize = 1)
        stitcher_ready = multiprocessing.Event()
        stitched = multiprocessing.Event()

        # MAKE SURE camP1 gets a left view of the scene, and camP2 gets the right view.
        camP1 = multiprocessing.Process(target = liveFeed, args=('https://192.168.1.206:8080/video', 
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

        print('All processes finished.')
    
    # OFFLINE MODE SELECTED.
    else:
        print('If video file is not in the script directory, enter the absolute path. Otherwise, just the file name would suffice.')
        print('NOTE: Enter '>' without commas to launch all the selected video files.')
        fpaths = []
        while True:
            p = input('Enter path or >: ')
            if p == '>':
                break
            else:
                fpaths.append(p) if os.path.exists(p) else print('[ERROR] Invalid path received. Ensure path/name is correct.')

        # Create as many empty elements as input paths in a list, and start a process
        # for each of the input video files in this list.
        vidProcesses = [[] for i in range(len(fpaths))]
        for i in range(len(vidProcesses)):
            vidProcesses[i] = multiprocessing.Process(target = videoPB, args = (fpaths[i],))
            print(vidProcesses[i])
            vidProcesses[i].start()
            
        for i in range(len(vidProcesses)):
            vidProcesses[i].join()
        
        print('[INFO] All videos have terminated. Exiting...')

# ******************* YOU HAVE REACHED THE END OF THE SCRIPT ******************* #