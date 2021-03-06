# Takes an image of random size as an input
# Resizes it to 1X224X224X3
# returns the prediction label
# 0 - green
# 1 - unknown
# 2 - red
# 3 - yellow

from styx_msgs.msg import TrafficLight
import numpy as np
import os
import sys
import tensorflow as tf
import cv2
from collections import defaultdict
from io import StringIO
import PIL.Image as Image
import PIL.ImageColor as ImageColor
import PIL.ImageDraw as ImageDraw
import PIL.ImageFont as ImageFont
import six
import rospy
import scipy.misc
from random import *

"""
		This is an inference class for Traffic light detection, It will load a trained
		graph with freezed weights and predicts the state of the light(14 classes)
		 such as red,yellow,green,redLeft,greenLeft, etc..
"""


class TLClassifierx(object):

    def __init__(self, *args):
        self.detection = -1
        model = 'light_classification/data/models/frcnn-x-6577.pb'
        self.detection_graph = tf.Graph()
        config = tf.ConfigProto()
        print("Graph loaded")
        config.gpu_options.allow_growth = True

        with self.detection_graph.as_default():
            od_graph_def = tf.GraphDef()

            with tf.gfile.GFile(model, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')
            self.sess = tf.Session(graph=self.detection_graph, config=config)

        self.input_img = self.detection_graph.get_tensor_by_name(
            'image_tensor:0')

        self.tl_classes = self.detection_graph.get_tensor_by_name(
            'detection_classes:0')

        self.tl_boxes = self.detection_graph.get_tensor_by_name(
            'detection_boxes:0')
        self.tl_scores = self.detection_graph.get_tensor_by_name(
            'detection_scores:0')
        self.num_detections = self.detection_graph.get_tensor_by_name(
            'num_detections:0')

        # from tensorflow object detection api
    def load_image_into_numpy_array(self, image):
        (rows, cols, channels) = image.shape
        (im_width, im_height) = (rows, cols)
        return np.array(image).reshape((im_height, im_width, 3)).astype(np.uint8)

    def draw_boxes(image_boxes_classes_scores):
        """Draws boxes on image."""
        (image, boxes, classes, scores) = image_boxes_classes_scores
        image_with_boxes = tf.py_func(visualize_boxes_fn,
                                      [image, boxes, classes, scores], tf.uint8)
        return image_with_boxes

    def get_classification(self, img):
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) - not required unless
        # reading input via cv2
        img = cv2.resize(img, (224, 224))
        img = np.expand_dims(img, axis=0)
        ###### Normalization code ########
        input_mean = 128
        input_std = 128
        img = img.astype(float)
        normalized = tf.divide(tf.subtract(img, [input_mean]), [input_std])
        sess = tf.Session()
        img_norm = sess.run(normalized)
        with self.detection_graph.as_default():
            (boxes, scores, classes, num) = self.sess.run(
                [self.tl_boxes, self.tl_scores,
                    self.tl_classes, self.num_detections],
                feed_dict={self.input_img: img_norm})



        self.detection = classes[0][0]
        if(classes[0][np.argmax(classes)] > 0.3):
            if self.detection == 0:
                rospy.logdebug('UNKNOWN ')
                return TrafficLight.UNKNOWN
            elif self.detection == 1:
                rospy.logdebug('GREEN')
                return TrafficLight.GREEN
            elif(self.detection == 2):
                rospy.logdebug('YELLOW')
                return TrafficLight.YELLOW
            elif(self.detection == 3):
                rospy.logdebug('RED')
                return TrafficLight.RED

        return TrafficLight.UNKNOWN

        return self.detection
