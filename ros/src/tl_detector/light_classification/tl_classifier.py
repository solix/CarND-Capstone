# Takes an image of random size as an input
# Resizes it to 1X224X224X3
# returns the prediction label
#0 - unknown
#1 - green
#2 - yellow
#3 - red

from styx_msgs.msg import TrafficLight
import numpy as np
import tensorflow as tf
import cv2
import rospy
import time

PRINT_DEBUG = False  # Print rospy.logwarn for debugging if True

class TLClassifier(object):

    def __init__(self, *args):
        self.detection = 0
        model = 'light_classification/intermediate_600.pb'
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
        #rospy.logwarn('########## CALLING CLASSIFIER #########')
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  #- not required unless
                                                    # reading input via cv2
        img = cv2.resize(img, (224, 224))
        img = np.expand_dims(img, axis=0)
        
        ###### Normalization code ########
        img_float = img.astype(float)
        image_mean = np.mean(img_float)
        img_norm = (img_float - image_mean) / image_mean
        
        with self.detection_graph.as_default():
            pred_class = self.sess.run(
                self.tl_class, feed_dict={self.input_img: img_norm})
            #float_formatter = lambda x: "%.2f" % x
            # rospy.logdebug("predicyion class: %s", pred_class)

        self.detection = np.argmax(pred_class)

        if(pred_class[0][np.argmax(pred_class)] > 0.3):
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

        return TrafficLight.UNKNOWN
