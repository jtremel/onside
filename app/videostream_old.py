import cv2
import pafy
import tensorflow as tf
import numpy as np
from app.model import label_map_util
import time

import sys

class VideoStream(object):
    
    sModelName = 'ssd_mobilenet_v1_coco_2017_11_17'
    sPathToFrozenGraph = 'app/model/' + sModelName + '/frozen_inference_graph.pb'
    sPathToLabels = 'app/model/mscoco_label_map.pbtxt'
    sModelName = "ssd_mobilenet_v1_coco_2018_01_28"
    
    sPluginDir = "/opt/intel/compute_vision_sdk/inference_engine/lib/ubuntu_18.04/intel64/"
    sPluginCpuExt = "/home/josh/inference_engine_samples_build/intel64/Release/lib/libcpu_extension.so"
    sTrackerName = "csrt"
    fThresholdScore = 0.6
    nMaxFramesDropped = 2
    
    nFrameRate = 30
    nResolutionWidth = 1280
    nResolutionHeight = 720

    oObjectTrackers = {
        "csrt": cv2.TrackerCSRT_create,
        "kcf": cv2.TrackerKCF_create,
        "mosse": cv2.TrackerMOSSE_create,
        "mil": cv2.TrackerMIL_create
    }

    _oTracker = None
    _oPlugin = None

    _boundingBox = None
    _oCategoryIndex = dict()
    _nFramesDropped = 0
    _bIsFirstFrame = True
    _nBallClassId = 37
    _oDetectionGraph = None
    _oVideo = None
    _nTotalFrames = 0
    _nTotalFramesDropped = 0
    _cachedSD = 0

    _bValidate = False

    def __init__(self, url):
        video = pafy.new(url)
        best = video.getbest(preftype="mp4")
        playurl = best.url

        self._oDetectionGraph = self._load_model(self.sPathToFrozenGraph)
        self._oCategoryIndex = self._load_category_map(self.sPathToLabels)
        self._oTracker = self.oObjectTrackers[self.sTrackerName]()

        self._oVideo = cv2.VideoCapture()
        self._oVideo.open(playurl)
        
    def process_stream(self):
        with self._oDetectionGraph.as_default():
            with tf.Session(graph=self._oDetectionGraph) as sess:
                while (self._oVideo.isOpened()):
                    bReadSuccess, frame = self._oVideo.read()

                    if bReadSuccess:
                        self._nTotalFrames += 1

                        if self._nTotalFrames > 1:
                            if self._detectCameraChange(frame):
                                self._bIsFirstFrame = True
                                self._boundingBox = None
                        else:
                            self.nResolutionHeight, self.nResolutionWidth = frame.shape[:2]

                        # Initial Detection
                        if self._bIsFirstFrame:

                            imageTensor = self._oDetectionGraph.get_tensor_by_name('image_tensor:0')
                            detectionBoxes = self._oDetectionGraph.get_tensor_by_name('detection_boxes:0')
                            detectionScores = self._oDetectionGraph.get_tensor_by_name('detection_scores:0')
                            detectionClasses = self._oDetectionGraph.get_tensor_by_name('detection_classes:0')
                            nDetections = self._oDetectionGraph.get_tensor_by_name('num_detections:0')

                            frameExpanded = np.expand_dims(frame, axis=0)

                            # Detect
                            (boxes, scores, classes, num) = sess.run(
                                [detectionBoxes, detectionScores, detectionClasses, nDetections],
                                feed_dict = {imageTensor: frameExpanded})
                           
                            classes = np.squeeze(classes).astype(np.int32)
                            boxes = np.squeeze(boxes)
                            scores = np.squeeze(scores)

                            # Filter categories
                            idx = np.unravel_index(
                                (classes == self._nBallClassId).argmax(), classes.shape)[0]
                            
                            if (idx == 0) or scores[idx] < self.fThresholdScore:
                                self._bIsFirstFrame = True
                                self._boundingBox = None
                            else:
                                boxes = boxes[idx]
                                self._boundingBox = self._get_bounding_box_coordinates(
                                    boxes,
                                    self.nResolutionWidth, self.nResolutionHeight,
                                    False
                                )
                                self._oTracker = self.oObjectTrackers[self.sTrackerName]()
                                self._oTracker.init(frame, self._boundingBox)
                                self._bIsFirstFrame = False
        
                        (H, W) = frame.shape[:2]

                        # check to see if we are currently tracking
                        if self._boundingBox is not None:
                            
                            (bTrackSuccess, box) = self._oTracker.update(frame)

                            if bTrackSuccess:
                                (x, y, w, h) = [int(v) for v in box]

                                # This is color and box drawing stuff here:
                                cv2.rectangle(frame, (x, y), (x+w, y+h),
                                    (66, 223, 244), 8)

                            # init info to display
                            #info = [
                            #    ("Method", self.sTrackerName),
                            #    ("Tracked", "Yes" if bTrackSuccess else "No")
                            #]
                            #
                            # draw stuff on the frame
                            #for (i, (k, v)) in enumerate(info):
                            #    text = "{}: {}".format(k,v)
                            #    cv2.putText(
                            #        frame, text, (10, H - ((i * 20) + 20)),
                            #        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2
                            #    )

                            if not bTrackSuccess:
                                self._nFramesDropped += 1

                            if self._nFramesDropped > self.nMaxFramesDropped:
                                self._bIsFirstFrame = True
                                self._nFramesDropped = 0
                            
                        # Show output frame
                        if self._bValidate:
                            if self._boundingBox is not None or not bTrackSuccess:
                                self._nTotalFramesDropped += 1

                            if self._nTotalFrames % 30 == 0:
                                print('Total Frames = {}'.format(self._nTotalFrames), file=sys.stdout)
                                print('Total Frames Dropped = {}'.format(self._nTotalFramesDropped), file = sys.stdout)

                        bEncodeSuccess, jpeg = cv2.imencode('.jpg', frame)
                        frame = jpeg.tobytes()
                        yield (b'--frame\r\n'
                        b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')


    def _detectCameraChange(self, image):
        mean, sd = cv2.meanStdDev(image)
        dif = sd[[0]] - self._cachedSD
        self._cachedSD = sd[[0]]
        if abs(dif) > 3:
            if self._bValidate:
                print('Camera Shot Change Detected! {}'.format(dif), file=sys.stdout)
            return True
        else:
            return False

    def streamRawVideo(self):
        while True:
            frame = self._get_frame()
            yield (b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

    def _get_frame(self):
        success, image = self._oVideo.read()
        ret, jpeg = cv2.imencode('.jpg', image)
        return jpeg.tobytes()
    
    def _get_bounding_box_coordinates(self, box, imgW, imgH, bSquare):
        xstart = box[1]*imgW
        ystart = box[0]*imgH
        width = box[3]*imgW - box[1]*imgW
        height = box[2]*imgH - box[0]*imgH
            
        if bSquare:
            width = min(width, height)
            height = width
        
        out = (xstart, ystart, width, height)
        return out

    def _load_model(self, path_to_frozen_graph):
        detection_graph = tf.Graph()
        with detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(path_to_frozen_graph, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')
        return detection_graph

    def _load_category_map(self, path_to_labels):
        categoryIndex = label_map_util.create_category_index_from_labelmap(
            path_to_labels, use_display_name=True)
        return categoryIndex

