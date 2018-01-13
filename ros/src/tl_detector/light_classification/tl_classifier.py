# Takes an image of random size as an input
# Resizes it to 1X224X224X3
# returns the prediction label
#0 - unknown
#1 - green
#2 - yellow
#3 - red

from styx_msgs.msg import TrafficLight
import numpy as np
import os
import sys
import tensorflow as tf
import cv2
from collections import defaultdict
from io import StringIO
import rospy
from matplotlib import pyplot as plt
from light_classification.utils import label_map_util
from light_classification.utils import visualization_utils as vis_util
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError


PRINT_DEBUG = True  # Print rospy.logwarn for debugging if True

if tf.__version__ < '1.4.0':
    raise ImportError(
        'Please upgrade your tensorflow installation to v1.4.* or later!')


class TLClassifier(object):

    def __init__(self, *args):
        self.detection = 0
        self.image_pub = rospy.Publisher(
            "/detector_node/image", Image, queue_size=1)
        self.bridge = CvBridge()

        self.MODEL_NAME = 'light_classification/data/models'
        # DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'
        # Path to frozen detection graph. This is the actual model that is used
        # for the object detection.
        self.PATH_TO_CKPT = self.MODEL_NAME + '/ssd-10183.pb'

        # List of the strings that is used to add correct label for each box.
        self.PATH_TO_LABELS = os.path.join(
            self.MODEL_NAME, 'label_bosch.pbtxt')

        self.NUM_CLASSES = 4

        # ## Load a (frozen) Tensorflow model into memory.

        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(self.PATH_TO_CKPT, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

        self.label_map = label_map_util.load_labelmap(self.PATH_TO_LABELS)
        self.categories = label_map_util.convert_label_map_to_categories(
            self.label_map, max_num_classes=self.NUM_CLASSES, use_display_name=True)
        self.category_index = label_map_util.create_category_index(
            self.categories)

    def get_classification(self, img):

        try:
                #### direct conversion to CV2 ####
            np_arr = np.fromstring(data.data, np.uint8)
            cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            ret = True
            # cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            ret = False
            print(e)
        (rows, cols, channels) = cv_image.shape
        if cols > 60 and rows > 60:
            ret = True

        with self.detection_graph.as_default():
            with tf.Session(graph=self.detection_graph) as sess:
                # Definite input and output Tensors for detection_graph
                image_tensor = self.detection_graph.get_tensor_by_name(
                    'image_tensor:0')
                # Each box represents a part of the image where a particular object was
                # detected.
                detection_boxes = self.detection_graph.get_tensor_by_name(
                    'detection_boxes:0')
                # Each score represent how level of confidence for each of the objects.
                # Score is shown on the result image, together with the class
                # label.
                detection_scores = self.detection_graph.get_tensor_by_name(
                    'detection_scores:0')
                detection_classes = self.detection_graph.get_tensor_by_name(
                    'detection_classes:0')
                num_detections = self.detection_graph.get_tensor_by_name(
                    'num_detections:0')
                while ret:
                    # image = Image.open(image_path)
                    # # the array based representation of the image will be used later in order to prepare the
                    # # result image with boxes and labels on it.
                    # image_np = load_image_into_numpy_array(image)
                    # Expand dimensions since the model expects images to have shape:
                    # [1, None, None, 3]
                    img_np = cv_image
                    image_np_expanded = np.expand_dims(img_np, axis=0)
                    # Actual detection.
                    (boxes, scores, classes, num) = sess.run(
                        [detection_boxes, detection_scores,
                            detection_classes, num_detections],
                        feed_dict={image_tensor: image_np_expanded})
                    # Visualization of the results of a detection.
                    vis_util.visualize_boxes_and_labels_on_image_array(
                        img_np,
                        np.squeeze(boxes),
                        np.squeeze(classes).astype(np.int32),
                        np.squeeze(scores),
                        self.category_index,
                        use_normalized_coordinates=True,
                        line_thickness=8)
                    # plt.figure(figsize=(12, 12))
                    # plt.imshow(img_np)
                    # plt.show()
                    # cv2.imshow('image', cv2.resize(img_np, (640, 488)))
                    #### Create CompressedIamge ####
                    msg = CompressedImage()
                    msg.header.stamp = rospy.Time.now()
                    msg.format = "jpeg"
                    msg.data = np.array(cv2.imencode(
                        '.jpg', img_np)[1]).tostring()
                    self.detection = np.argmax(classes)

                    try:
                        self.image_pub.publish(
                            self.bridge.cv2_to_imgmsg(img_np, "bgr8"))
                        # cv2.imwrite('res/' + str(msg.header.stamp) +'camera_image.jpeg', img_np)
                        rospy.loginfo("processed the IMage")
                        ret = False
                        if(classes[0][np.argmax(classes)] > 0.3):
                            if self.detection == 0:
                                if PRINT_DEBUG:
                                    rospy.logwarn('UNKNOWN ')
                                return TrafficLight.UNKNOWN
                            elif self.detection == 1:
                                if PRINT_DEBUG:
                                    rospy.logwarn('GREEN')
                                return TrafficLight.GREEN
                            elif(self.detection == 2):
                                if PRINT_DEBUG:
                                    rospy.logwarn('YELLOW')
                                return TrafficLight.YELLOW
                            elif(self.detection == 3):
                                if PRINT_DEBUG:
                                    rospy.logwarn('RED')
                                return TrafficLight.RED
                    except CvBridgeError as e:
                        print(e)

        

        return TrafficLight.UNKNOWN
