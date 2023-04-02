Project Members: Syed Zain Abbas, Taimoor Hasan Khan (maintainer), Yaseen Athar

https://user-images.githubusercontent.com/94681976/226663837-897862a2-5f77-41ee-ba1a-beee7f420091.mp4

Welcome to the PoolBallTracking-CV wiki! This repo is a semester project titled "Multi-Camera Surveillance System" for the Computer Vision course at NUST. Our scenario is not surveillance, but the core idea is the same. 

We want to track the movement of pool balls in a pool game using a stitched view from at least two or more cameras. Such a system could be used to keep track of the game score, among other things. However, the main goal is to become familiar with coding the concepts learned over the course and the typical environment and modules we can use.

# Task List
The project was split into **SIX** tasks, of which **ONE** is optional.

1. Multi-Camera Setup and Recording (with offline and online playback support).
2. Object Detection using YOLO Object Detector (the object being the pool ball).
3. Orthographic Top-View Generation.
4. Visualisation of Object Detection on Stitched Orthographic Top-Views of the cameras.
5. Heatmap Visualisation on Orthographic Top-View.
6. (Optional) Trip Wire feature.
   

### General Method and Challenges in Each Task

**Task 1:** We used the IP Webcam app, as using IP cameras was a requirement in this step. We used a multiprocessing framework to reduce lag as much as possible. 

- **Challenge:** While the framework works excellently for USB-connected cameras, there is still a noticeable buffer lag on IP cameras. As far as we could research, this was an OpenCV problem with how `cv2.imread` handled buffers. Perhaps the `imutils` module's reader would be better as it uses threaded implementations.

**Task 2:** YOLOv4 was the latest at the time. We used **AlexeyAB's** [**darknet**](https://github.com/AlexeyAB/darknet) implementation. For training on a set of pool balls, we scraped the internet for images of pool balls using **skaldek's** [**fork**](https://github.com/skaldek/ImageNet-Datasets-Downloader) of **mf1024's** [**ImageNet-Datasets-Downloader**](https://github.com/mf1024/ImageNet-Datasets-Downloader), manually removed bad samples, and annotated the rest using [**LabelImg**](https://github.com/heartexlabs/labelImg).

<p align="center">
    <img src=https://user-images.githubusercontent.com/94681976/226643363-f8eb7090-b364-444e-ae3d-eb5db2615744.jpg />
</p>

***

### ***PROJECT WAS COMPLETED APART FROM THE OPTIONAL TRIPWIRE FEATURE***

Though we did not perform the optional task owing to time constraints, a naive and simple approach would be to draw a small box around the pockets and check when a pool ball's bounding box center passes that boundary. Since we are in the top view, this is a fairly accurate measure of when a ball is pocketed. In this context, a trip wire could determine when a ball is pocketed and keep track of the score.

##### Further Improvements
- Automating the process of top orthographic view generation. We could try to detect the table corners in the stitched view or fit a plane that maximized the area of the green surface. Then we have a pretty good guess of where the corners will be.
- Automating pocket detection to draw tripwires quickly.
- Adding proper tracker IDs to each ball and code checks for missed detections in frames.
- Gathering a more diverse dataset will make the model robust to various lighting conditions and pool ball orientations.
