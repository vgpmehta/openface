import os, sys
CURRENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))+'/build/exe/FeatureExtractionPython'
print(CURRENT_DIR)
sys.path.append(CURRENT_DIR)

import time
#import keyboard
from libPyOpenfaceVideo import *

ov = libPyOpenfaceVideo.OpenfaceVideo()
ov.start()

while( ov.is_running() ):
	
	#if keyboard.is_pressed('q'):
	#	ov.stop()
	
	if ov.has_frame():
		print( ov.get_frame() )
	
	time.sleep( 0.01 )