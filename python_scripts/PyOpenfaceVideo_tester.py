import os, sys
OFV_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))+'/build/exe/FeatureExtractionPython'
sys.path.append(OFV_DIR)

import time
import keyboard
from PyOpenfaceVideo import *

ov = OpenfaceVideo()

ov.device = 0

# set model path: mandatory to start tracking
ov.landmark_model = OFV_DIR + "/model/main_ceclm_general.txt"
ov.HAAR = OFV_DIR + "/classifiers/haarcascade_frontalface_alt.xml"
ov.MTCNN = OFV_DIR + "/model/mtcnn_detector/MTCNN_detector.txt"

# start tracking
ov.start()

while( ov.is_running() ):
	
	try:
		if keyboard.is_pressed('q'):
			ov.stop()
	except:
		pass
		
	if ov.has_frame():
		print( "new frame: ", ov.frame.ID )
		print( "\tgaze_0: ", ov.frame.gaze_0.x, ',', ov.frame.gaze_0.y, ',', ov.frame.gaze_0.z )
		print( "\tgaze_1: ", ov.frame.gaze_1.x, ',', ov.frame.gaze_1.y, ',', ov.frame.gaze_1.z )
		print( "\thead_rot: ", ov.frame.head_rot.x, ',', ov.frame.head_rot.y, ',', ov.frame.head_rot.z )
		print( "\thead_pos: ", ov.frame.head_pos.x, ',', ov.frame.head_pos.y, ',', ov.frame.head_pos.z )
		print( "\tlandmarks: " )
		for i in range( len(ov.frame.landmarks) ):
			print( "\t\t", i, ':', ov.frame.landmarks[i].x, ',', ov.frame.landmarks[i].y, ',', ov.frame.landmarks[i].z )
	
	time.sleep( 0.01 )