import cv2
import pafy

class VideoStream(object):
        
    urlname = None
    sModelName = 'ssd_mobilenet_v1_coco_2017_11_17'
    sPathToFrozenGraph = sModelName + '/frozen_inference_graph.pb'
    sPathToLabels = 'model/mscoco_label_map.pbtxt'
    sTrackerName = 'csrt'
    nBallClassId = 37
    fThresholdScore = 0.6
    nMaxFramesDropped = 10
    nFrameRate = 30
    oResolution = (1280, 720)

    def __init__(self,url):
        urlname = url
        video = pafy.new(url)
        best = video.getbest(preftype="mp4")
        playurl = best.url

        self.video = cv2.VideoCapture()
        self.video.open(playurl)


    def __del__(self):
        self.video.release()


    def get_frame(self):
        success, image = self.video.read()
        return image
    
    def get_tracked_frame(self, image):
        ret, jpeg = cv2.imencode('.jpg', image)
        return jpeg.tobytes()

    def check_box_bounds(self, box):
        if any(box < 0.05):
            return False
        elif any(box > 0.95):
            return False
        else:
            return True
    
    def get_bounding_box_coordinates(self, box, imgW, imgH, bSquare):
        xstart = box[1]*imgW
        ystart = box[0]*imgH
        width = box[3]*imgW - box[1]*imgW
        height = box[2]*imgH - box[0]*imgH
            
        if bSquare:
            width = min(width, height)
            height = width
        
        out = (xstart, ystart, width, height)
        return out

    def load_model(self):
        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(self.sPathToFrozenGraph, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')