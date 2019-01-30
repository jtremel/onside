#import os
#import six.moves.urllib as urllib
#import sys
import numpy as np
import time
import cv2
import imutils
from imutils.video import VideoStream
from imutils.video import FPS

import tensorflow as tf
from app import label_map_util # this is from tensorflow

#from collections import defaultdict
#from io import StringIO
#from matplotlib import pyplot as plt
#from PIL import Image

#~~~~~~ VARIOUS SETTINGS ~~~~~~~~#
# What model to download.
MODEL_NAME = 'ssd_mobilenet_v1_coco_2017_11_17'
# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_FROZEN_GRAPH = 'model/' + MODEL_NAME + '/frozen_inference_graph.pb'
# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = 'model/mscoco_label_map.pbtxt'

# Size, in inches, of the output images.
IMAGE_SIZE = (30, 25)

ball_class_id = 37

inputFilename="/home/josh/Downloads/video_examples/soccer_test_03.mp4"
threshold_score = 0.6

#outputFilename='/home/josh/Downloads/video_examples/soccer_test_03__tracked2.mp4'
#codec = cv2.VideoWriter_fourcc('m','p','4','v')

framerate=30
resolution=(1280,720)

trackerName = "csrt"
nMaxFramesDropped = 10
#~~~~~~END SETTINGS ~~~~~~~~#

def check_box_bounds(box):
    if any(box < 0.05):
        return False
    elif any(box > 0.95):
        return False
    else:
        return True
    
def get_bounding_box_coordinates(box, imgW, imgH, bSquare):
    xstart = box[1]*imgW
    ystart = box[0]*imgH
    width = box[3]*imgW - box[1]*imgW
    height = box[2]*imgH - box[0]*imgH
    
    if bSquare:
        width = min(width, height)
        height = width
    
    out = (xstart, ystart, width, height)
    return out




# Grab the model graph
detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')

# Set up detection categories
category_index = label_map_util.create_category_index_from_labelmap(
    PATH_TO_LABELS, use_display_name=True)

# initialize a dictionary that maps strings to their corresponding
# OpenCV object tracker implementations
OPENCV_OBJECT_TRACKERS = {
    "csrt": cv2.TrackerCSRT_create,
    "kcf": cv2.TrackerKCF_create,
    "mosse": cv2.TrackerMOSSE_create
}

# grab the appropriate object tracker using our dictionary of
# OpenCV object tracker objects
tracker = OPENCV_OBJECT_TRACKERS[trackerName]()
    


# initialize the bounding box coordinates of the object to track
initBB = None

nFramesDropped = 0
isFirstFrame = True

vidCapture = cv2.VideoCapture(inputFilename)
#videoFileOutput = cv2.VideoWriter(outputFilename, codec, framerate, resolution)

with detection_graph.as_default():
    with tf.Session(graph=detection_graph) as sess:
        
        # loop over frames from the video stream
        while(vidCapture.isOpened()):
            
            ret, frame = vidCapture.read()
            
            if ret == True:
                
                # initial detection
                if isFirstFrame:
                    imageTensor = detection_graph.get_tensor_by_name('image_tensor:0')
                    detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
                    detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
                    detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
                    num_detections = detection_graph.get_tensor_by_name('num_detections:0')
                    
                    frame_expanded = np.expand_dims(frame, axis=0)
                    
                    # detection
                    (boxes, scores, classes, num) = sess.run([detection_boxes,
                                                             detection_scores,
                                                             detection_classes,
                                                             num_detections],
                                        feed_dict = {imageTensor: frame_expanded})
                    
                    classes = np.squeeze(classes).astype(np.int32)
                    boxes = np.squeeze(boxes)
                    scores = np.squeeze(scores)
                    
                    detectionBoxes = []
                    detectionClasses = []
                    detectionScores = []
                    for item in enumerate(classes):
                        if ( item[1] == ball_class_id and 
                            scores[item[0]] > threshold_score and
                            check_box_bounds(boxes[item[0]]) 
                           ):
                            detectionBoxes.append(boxes[item[0]])
                            detectionClasses.append(classes[item[0]])
                            detectionScores.append(scores[item[0]])
                            
                            #take only the highest confidence object
                            break
                            
                    boxes = np.asarray(detectionBoxes)
                    classes = np.asarray(detectionClasses)
                    scores = np.asarray(detectionScores)
                                        
                    if len(boxes) == 0:
                        isFirstFrame = True
                        initBB = None
                    else:
                        initBB = get_bounding_box_coordinates(boxes[0],resolution[0],resolution[1], False)
                        tracker = OPENCV_OBJECT_TRACKERS[trackerName]()
                        tracker.init(frame, initBB)
                        isFirstFrame = False


                #frame = imutils.resize(frame, width=500)
                (H, W) = frame.shape[:2]

                # check to see if we are currently tracking an object
                if initBB is not None:
                    # grab the new bounding box coordinates of the object
                    (success, box) = tracker.update(frame)

                    # check to see if the tracking was a success
                    if success:
                        (x, y, w, h) = [int(v) for v in box]
                        cv2.rectangle(frame, (x, y), (x + w, y + h),
                            (255, 0, 0), 3)

                    # initialize the set of information we'll be displaying on
                    # the frame
                    info = [
                        ("Method", "CSRT"),
                        ("Tracked?", "Yes" if success else "No"),
                    ]

                    # loop over the info tuples and draw them on our frame
                    for (i, (k, v)) in enumerate(info):
                        text = "{}: {}".format(k, v)
                        cv2.putText(frame, text, (10, H - ((i * 20) + 20)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

                    if not success:
                        nFramesDropped += 1

                    if nFramesDropped > nMaxFramesDropped:
                        isFirstFrame = True
                        nFramesDropped = 0

                
                # write video file
                #videoFileOutput.write(frame)
                
                # show the output frame
                cv2.imshow("Live detection", frame)
                
                # process keystrokes
                key = cv2.waitKey(1) & 0xFF

                # pressing "s" allows user to draw bounding box
                if key == ord("s"):
                    # select the bounding box of the object we want to track (make
                    # sure you press ENTER or SPACE after selecting the ROI)
                    initBB = cv2.selectROI("Frame", frame, fromCenter=False,
                        showCrosshair=True)
                    print(initBB)
                    # start OpenCV object tracker using the supplied bounding box
                    # coordinates, then start the FPS throughput estimator as well
                    tracker.init(frame, initBB)

                # if the `q` key was pressed, break from the loop
                elif key == ord("q"):
                    vidCapture.release()
                    #videoFileOutput.release()
                    cv2.destroyAllWindows()
                    break
                    
            else:
                vidCapture.release()
                #videoFileOutput.release()
                cv2.destroyAllWindows()
