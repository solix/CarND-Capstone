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


class TLClassifier(object):

    def __init__(self, *args):
        self.detection = 0
        model = 'light_classification/rounded_graph.pb'
        self.detection_graph = tf.Graph()
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        with self.detection_graph.as_default():
            od_graph_def = tf.GraphDef()

            with tf.gfile.GFile(model, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')
            self.sess = tf.Session(graph=self.detection_graph, config=config)

        self.input_img = self.detection_graph.get_tensor_by_name('input:0')
        self.tl_class = self.detection_graph.get_tensor_by_name(
            'final_result:0')

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
        ######## Normalization code - End ########
        with self.detection_graph.as_default():
            pred_class = self.sess.run(
                self.tl_class, feed_dict={self.input_img: img_norm})
            float_formatter = lambda x: "%.2f" % x
            # rospy.logdebug("predicyion class: %s", pred_class)

        self.detection = np.argmax(pred_class)

        if(pred_class[0][np.argmax(pred_class)] > 0.6):

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
