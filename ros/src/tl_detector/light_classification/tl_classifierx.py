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
		model = './light_classification/data/models/sim_retrained_mobilenet.pb'
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
		self.tl_class = self.detection_graph.get_tensor_by_name('final_result:0')

	
	def get_classification(self,img):
		
		img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
		img = cv2.resize(img,(224,224))
		img = np.expand_dims(img,axis=0)
		with self.detection_graph.as_default():
			pred_class = self.sess.run(self.tl_class,feed_dict={self.input_img: img})
		self.detection = np.argmax(pred_class)
		return self.detection

