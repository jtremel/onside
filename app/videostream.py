import cv2
import pafy
import tensorflow as tf
import numpy as np
from app.model import label_map_util

class VideoStream(object):
        
    sUrlname = None
    sModelName = 'ssd_mobilenet_v1_coco_2017_11_17'
    sPathToFrozenGraph = 'app/model/' + sModelName + '/frozen_inference_graph.pb'
    sPathToLabels = 'app/model/mscoco_label_map.pbtxt'
    sTrackerName = "csrt"
    fThresholdScore = 0.6
    nMaxFramesDropped = 10
    nFrameRate = 30
    oResolution = (1280, 720)

    oObjectTrackers = {
        "csrt": cv2.TrackerCSRT_create,
        "kcf": cv2.TrackerKCF_create,
        "mosse": cv2.TrackerMOSSE_create
    }

    _oTracker = None
    _boundingBox = None
    _oCategoryIndex = dict()
    _nFramesDropped = 0
    _bIsFirstFrame = True
    _nBallClassId = 37
    _oDetectionGraph = None
    _oVideo = None
    
    def __init__(self,url):
        sUrlname = url
        video = pafy.new(url)
        best = video.getbest(preftype="mp4")
        playurl = best.url

        self._oDetectionGraph = self._load_model(self.sPathToFrozenGraph)
        self._oCategoryIndex = self._load_category_map(self.sPathToLabels)
        self._oTracker = self.oObjectTrackers[self.sTrackerName]()

        self._oVideo = cv2.VideoCapture()
        self._oVideo.open(playurl)


    def __del__(self):
        self._oVideo.release()

    def process(self):
        while True:
            frame = self._get_frame()
            yield (b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')


    def process_stream(self):
        with self._oDetectionGraph.as_default():
            with tf.Session(graph=self._oDetectionGraph) as sess:
                while (self._oVideo.isOpened()):

                    bReadSuccess, frame = self._oVideo.read()

                    if bReadSuccess:

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
                                [detectionBoxes,
                                detectionScores,
                                detectionClasses,
                                nDetections],
                                feed_dict = {imageTensor: frameExpanded})
                            
                            classes = np.squeeze(classes).astype(np.int32)
                            boxes = np.squeeze(boxes)
                            scores = np.squeeze(scores)

                            detectionBoxesOfInterest = []
                            detectionClassesOfInterst = []
                            detectionScoresOfInterest = []
                            for item in enumerate(classes):
                                if ( item[1] == self._nBallClassId and
                                    scores[item[0]] > self.fThresholdScore and
                                    self._check_box_bounds(boxes[item[0]])
                                ):
                                    detectionBoxesOfInterest.append(boxes[item[0]])
                                    detectionClassesOfInterst.append(classes[item[0]])
                                    detectionScoresOfInterest.append(scores[item[0]])

                                    # Take only the highest confidence object
                                    break

                            boxes = np.asarray(detectionBoxesOfInterest)
                            classes = np.asarray(detectionClassesOfInterst)
                            scores = np.asarray(detectionScoresOfInterest)

                            if len(boxes) == 0:
                                self._bIsFirstFrame = True
                                self._boundingBox = None
                            else:
                                self._boundingBox = self._get_bounding_box_coordinates(
                                    boxes[0],
                                    self.oResolution[0],
                                    self.oResolution[1],
                                    False
                                )
                                self._oTracker = self.oObjectTrackers[self.sTrackerName]()
                                self._oTracker.init(frame, self._boundingBox)
                                self._bIsFirstFrame = False

                        # resize frame if needed: frame = imutils.resize(frame, width =500)
                        (H, W) = frame.shape[:2]

                        # check to see if we are currently tracking
                        if self._boundingBox is not None:
                            (bTrackSuccess, box) = self._oTracker.update(frame)

                            if bTrackSuccess:
                                (x, y, w, h) = [int(v) for v in box]

                                # This is color and box drawing stuff here:
                                cv2.rectangle(frame, (x, y), (x+w, y+h),
                                    (255, 0, 0), 3)

                            # init info to display
                            info = [
                                ("Method", self.sTrackerName),
                                ("Tracked", "Yes" if bTrackSuccess else "No")
                            ]

                            # draw stuff on the frame
                            for (i, (k, v)) in enumerate(info):
                                text = "{}: {}".format(k,v)
                                cv2.putText(
                                    frame,
                                    text,
                                    (10, H - ((i * 20) + 20)),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.6,
                                    (0,255,0), 
                                    2
                                )
                            
                            if not bTrackSuccess:
                                self._nFramesDropped += 1

                            if self._nFramesDropped > self.nMaxFramesDropped:
                                self._bIsFirstFrame = True
                                self._nFramesDropped = 0
                            
                        # Show output frame
                        #cv2.imshow("Live Detection", frame)
                        bEncodeSuccess, jpeg = cv2.imencode('.jpg', frame)
                        frame = jpeg.tobytes()
                        yield (b'--frame\r\n'
                        b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')


#                    else:
#                        vidCapture.release()
#                        cv2.destroyAllWindows()



    
    def _get_frame(self):
        success, image = self._oVideo.read()
        ret, jpeg = cv2.imencode('.jpg', image)
        return jpeg.tobytes()
    
    def _check_box_bounds(self, box):
        if any(box < 0.05):
            return False
        elif any(box > 0.95):
            return False
        else:
            return True
    
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

