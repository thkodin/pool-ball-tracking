# Check orthographic top view generation and heatmap building independently.
import cv2 as cv
import numpy as np
import queue

# Function to mark 4 points on the planar object to compute bird's eye view
# homography for warping. This is done only once with the assumption that the
# cameras do not move throughout the stream.
def markPoints(event, x, y, flags, param):
    if event == cv.EVENT_LBUTTONDOWN:                       # on left-click (downpress only)
        cv.circle(fframe, (x, y), 5, (150, 150, 255), -1)   # help visualize marks

        if len(points) < 4:     
            points.append([x, y])
        elif len(points) == 4:
            cv.destroyAllWindows()
            print("[INFO] Marked 4 points...")
        cv.imshow("Mark 4 Points (TL, TR, BL, BR)", fframe)

vid = cv.VideoCapture("StitchedView_1.avi")

# Threshold will need a bit of tweaking now and then. For better robustness, use
# the morphological expansions.
bgSub = cv.createBackgroundSubtractorMOG2(varThreshold=500, detectShadows=False)
hmap_historyInSecs = 0.1

# You can use a list method to remove the first entry in history as well, but queues are 
# cleaner and easier to maintain.
krnl = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))          # for filtering noise
queue_hmap = queue.Queue(maxsize = hmap_historyInSecs * 30)     # assuming 30 FPS
first_frame = True                                              # do first frame stuff
points = []                                                     # store points for homography
homography_top = None

while True:
    grabbed, frame = vid.read()
    if grabbed and first_frame is True:
        fframe = frame.copy()
        cv.imshow("Mark 4 Points (TL, TR, BL, BR)", fframe)
        cv.setMouseCallback("Mark 4 Points (TL, TR, BL, BR)", markPoints)
        cv.waitKey(0)

        h, w =  points[2][1] - points[0][1], points[1][0] - points[0][0]    # height, width 

        points = np.float32(np.asanyarray(points))
        points_top = np.array(([0, 0], [w, 0], [0, h], [w, h]), np.float32)

        homography_top = cv.getPerspectiveTransform(points, points_top)
        accum_image = np.zeros((h, w), np.uint8)

        first_frame = False

    elif grabbed is False:
        print("[INFO] Video has ended.")
        break
    
    # Remember, warping takes output size in (width, height) order.
    frame = cv.warpPerspective(frame, homography_top, (w, h))

    # Subtract if history queue full. This will store the latest results as
    # defined earlier.
    if queue_hmap.full():
        accum_image = cv.subtract(accum_image, queue_hmap.get())
    
    bgSubbed = bgSub.apply(frame)   # returns a binary image of the background

    # Use if MOG threshold doesn't cut it. MorphEx is just a fancy name for erosion
    # dilation series to help remove noise (small white spots get eroded away, and actual
    # objects are dilated back to original shape).
    bgSubbed_lessNoise = cv.morphologyEx(bgSubbed, cv.MORPH_OPEN, krnl)     

    # Create a binary threshold to filter out noise from the model.
    _, th1 = cv.threshold(bgSubbed, 250, 100, cv.THRESH_BINARY)

    # Accumulate the heatmap image every frame.
    accum_image = cv.add(accum_image, th1)
  
    # Add the filter threshold to history queue.
    queue_hmap.put(th1)

    # Provide a colormap to the masked image.
    heatmapped = cv.applyColorMap(accum_image, cv.COLORMAP_SUMMER)

    #cv.imshow("filt", bgSubbed)
    #cv.imshow("Mask Only", th1)
    cv.imshow("Original", frame)
    cv.imshow("Heatmap", heatmapped)
    key = cv.waitKey(30)
    if key == ord('q'):
        break

vid.release()
cv.destroyAllWindows()