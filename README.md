Project Members: Syed Zain Abbas, Taimoor Hasan Khan (maintainer), Yaseen Athar

https://user-images.githubusercontent.com/94681976/226663837-897862a2-5f77-41ee-ba1a-beee7f420091.mp4

We want to track the movement of pool balls in a pool game using a stitched view from at least two or more cameras. Such a system could be used to keep track of the game score, among other things. However, the main goal is to become familiar with coding the concepts learned over the course and the typical environment and modules we can use.

# Task List
The project was split into **SIX** tasks, of which **ONE** is optional.

1. Multi-Camera Setup and Recording (with offline and online playback support).
2. Object Detection using YOLO Object Detector (the object being the pool ball).
3. Orthographic Top-View Generation.
4. Visualisation of Object Detection on Stitched Orthographic Top-Views of the cameras.
5. Heatmap Visualisation on Orthographic Top-View.
6. (Optional) Trip Wire feature.
   

### Task 1

We used the IP Webcam app, as using IP cameras was a requirement in this step. We used a multiprocessing framework to reduce lag as much as possible. 

- **Challenge:** While the framework works excellently for USB-connected cameras, there is still a noticeable buffer lag on IP cameras. As far as we could research, this was an OpenCV problem with how `cv2.imread` handled buffers. Perhaps the `imutils` module's reader would be better as it uses threaded implementations.

### Task 2

YOLOv4 was the latest at the time. We used *AlexeyAB's* [**darknet**](https://github.com/AlexeyAB/darknet) implementation of YoloV4-Tiny for video inference at a tested average of 30 FPS (full model gave around 10-12 FPS). 

For training on a set of pool balls, we scraped the internet for images of pool balls using *skaldek's* [**fork**](https://github.com/skaldek/ImageNet-Datasets-Downloader) of *mf1024's* [**ImageNet-Datasets-Downloader**](https://github.com/mf1024/ImageNet-Datasets-Downloader), manually removed bad samples, and annotated the rest using [**LabelImg**](https://github.com/heartexlabs/labelImg). Final set had a total of **210** images. Example training image:

<p align="center">
    <img src=https://user-images.githubusercontent.com/94681976/226643363-f8eb7090-b364-444e-ae3d-eb5db2615744.jpg title="Example Training Image" />
</p>

Training parameters were:
- lasses         = 1 ("pool ball")
- Batch Size      = 64
- Mini Batches    = 16
- Max Iterations  = 2000
- Blob Size       = (416, 416)

The training graph (as presented by Darknet) is as follows:

<p align="center">
    <img src=https://user-images.githubusercontent.com/94681976/229373370-72e33131-2e7d-4d3e-a7dd-7e6c0bc46f66.png title="Example Training Image" width=50% />
</p>

### Task 3

This step was quite simple. For our initial test, we setup a single camera to view the entire pool table. We manually marked four corners of the pool table (top left, top right, bottom right, bottom left), and fed those points to `cv2.getPerspectiveTransform` to estimate the 2D homography (requires 8 points - each corner we mark gets us a pair of points, so 4 corners = 2 x 4 = 8 points). Then, we do `cv2.warpPerspective` to apply the homography to the image, which gets us the top or bird's eye view.

We extend this to the stitched view of two cameras instead of just one in Task 4.

### Task 4

> IMPORTANT: This step requires use of the OpenCV's Features2D framework. We additionally require access to SIFT, BRISK, ORB, or even SURF features. Ensure your OpenCV installation has access to the `opencv-contrib` module that contains several extra and 'non-free for commerical use' features. For more info, see about building OpenCV from source: [**official instructions**](https://docs.opencv.org/4.x/d5/de5/tutorial_py_setup_in_windows.html).

This step involves image stitching. We use two IP cameras from the multiprocessing framework setup in Task 1 - each camera views the left and right side of the table, ~70% of the full pool table. We then extract the SIFT features from each view and brute-force match them (can further improve match accuracy via fundamental matrix estimation and RANSAC outlier elimination). Once the points are matched, we estimate the homography using `cv2.getHomography`, which is an n-point version of the 4-point `cv2.getPerspectiveTransform` method. Then we warp the right camera to fit the view of the left one to get the stitched result as a new image. 

As an additional step, we apply some refinements to minimize the black padding spaces around the stitched result, and also provide an option to get rid of all the black space, though that may cause information loss so we do not recommend using it.

We can now generate a top view of the stithced result showing the full pool table. The process remains as described in Task 3. Finally, we apply YOLO to the stitched top view to detect all the pool balls.

### Task 5


***

### ***PROJECT WAS COMPLETED APART FROM THE OPTIONAL TRIPWIRE FEATURE***

Though we did not perform the optional task owing to time constraints, a naive and simple approach would be to draw a small box around the pockets and check when a pool ball's bounding box center passes that boundary. Since we are in the top view, this is a fairly accurate measure of when a ball is pocketed. In this context, a trip wire could determine when a ball is pocketed and keep track of the score.

##### Further Improvements and Addons

Training:
- Gathering a larger and more diverse dataset will make the model robust to various lighting conditions and pool ball orientations.
- Add control samples that contain circular objects other than pool balls. Right now, the network will call any circular, ball-like object a pool ball.

Automation:
- Automating the process of top orthographic view generation. We could try to detect the table corners in the stitched view or fit a plane that maximized the area of the green surface. Then we have a pretty good guess of where the corners will be.
- Automating pocket detection to draw tripwires quickly.

Heatmap:
- Skeletonize (thin via erosion morphing) the heatmap and lower the history. Right now, we get colors the size of the entire object and they stay for a long time before merging back into the background. That looks quite ugly and unclear at times.

Tracking:
- Implement a tracking algorithm (e.g., KCF) using the pool ball detections by YOLO as initial locations. We won't need to run YOLO on each frame that way, improving performance. Simply run YOLO until all pool balls are detected, and then run the tracking algorithm for successive frames. Keep checks checks for tracking failure - reset tracker and run YOLO again if it happens.
