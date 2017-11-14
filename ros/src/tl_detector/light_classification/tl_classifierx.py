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
import viz_util
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
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_np = self.load_image_into_numpy_array(img)
        img_np = np.expand_dims(img_np, axis=0)
        with self.detection_graph.as_default():
            (boxes, scores, classes, num) = self.sess.run(
                [self.tl_boxes, self.tl_scores,
                    self.tl_classes, self.num_detections],
                feed_dict={self.input_img: img_np})

        rospy.logdebug("[light] box found: %s ", boxes[0][0])
        rospy.logdebug("[light] classes found: %s ", classes[0][0])

        res = cv2.rectangle(img, (boxes[0][0, 0], boxes[0][0, 1]), (boxes[
                            0][0, 2] - boxes[0][0, 0], boxes[0][0, 3] - boxes[0][0, 1]), (255, 0, 0), 5)
        scipy.misc.imsave('./light_classification/data/res/out' +
                          str(classes[0][0]) + "-" + str(randint(0, 10)) + '.jpg', res)

        self.detection = classes[0][0]
        # print("classified number : {} " .format(classes))

        return self.detection

    # def classify_viz():
