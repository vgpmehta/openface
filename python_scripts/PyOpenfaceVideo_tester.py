import os, sys
OFV_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))+'/build/exe/FeatureExtractionPython'
sys.path.append(OFV_DIR)

import time
import keyboard
from PyOpenfaceVideo import *

def frame_parsing( frame ):
	print( '>>>> ', frame['ID'] )
	print( frame['gaze_0'] )
	print( frame['gaze_1'] )
	print( frame['head_rot'] )
	print( frame['head_pos'] )
	print( frame['landmarks'] )

ov = OpenfaceVideo()

ov.device = 0

# set model path: mandatory to start tracking
ov.landmark_model = OFV_DIR + "/model/main_ceclm_general.txt"
ov.HAAR = OFV_DIR + "/classifiers/haarcascade_frontalface_alt.xml"
ov.MTCNN = OFV_DIR + "/model/mtcnn_detector/MTCNN_detector.txt"

# setting the callback function
ov.callback_frame( frame_parsing )

# start tracking
ov.start()