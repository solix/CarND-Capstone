#Takes an image of random size as an input
#Resizes it to 1X224X224X3
#returns the prediction label
#0 - green
#1 - unknown
#2 - red
#3 - yellow

from styx_msgs.msg import TrafficLight
import numpy as np
import os
import sys
import tensorflow as tf
import cv2
from collections import defaultdict
from io import StringIO

class TLClassifierx(object):

	def __init__(self,*args):
		self.detection = 0
		model = 'light_classification/data/models/frcnn-i-11890.pb'
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

		self.input_img = self.detection_graph.get_tensor_by_name('image_tensor:0')

		self.tl_class = self.detection_graph.get_tensor_by_name('detection_classes:0')

		self.tl_boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
		self.tl_scores = self.detection_graph.get_tensor_by_name('detection_scores:0')

	
	def get_classification(self,img):
		img_np = load_image_into_numpy_array(img)
		img_np = np.expand_dims(img_np,axis=0)
		with self.detection_graph.as_default():
			pred_class = self.sess.run(self.tl_class,feed_dict={self.input_img: img_np})
		self.detection = np.argmax(pred_class)
		return self.detection

	#from tensorflow object detection api
	def load_image_into_numpy_array(image):
		(im_width, im_height) = image.size
		return np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)	

