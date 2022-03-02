import zmq
import pandas as pd
import threading
import platform
import os
from pathlib import Path

def launch_live_openFace(port):
        """Helper method to run openface on a file

        Returns:
            [exit_status]: [exit_status]
        """
        plt = "Windows" if platform.system()== "win32" else "linux"
        root_dir =   os.path.dirname(Path(__file__).absolute())
        print("root_dir : "+str(root_dir))
        openFace_path = os.path.join(root_dir,"..\\..\\build\\bin\\AuExtractionMQ.exe") if plt=="Windows" else os.path.join(root_dir,"../../build/bin/AuExtractionMQ")
        print("openFace_path : "+str(openFace_path))
        cmd = str(f"%s  -device 0 -zmq_port %s"%(openFace_path, port) )
        cmd = 'start /W '+cmd if plt=="Windows" else cmd
        print(">> process_openFace cmd : "+str(cmd))
        exit_status = os.system(cmd)
        return exit_status

ctx = zmq.Context.instance()
socket = ctx.socket(zmq.SUB)
port = 5556
socket.connect('tcp://localhost:%s' % port)
socket.subscribe = ''
au_df = pd.DataFrame()
# launch OpenFace live  Aus extraction
thr = threading.Thread(target=launch_live_openFace, args=([port]), kwargs={})
thr.start()
while thr.is_alive():
    if socket.poll(0) == zmq.POLLIN:
        msg =socket.recv()
        aus = ast.literal_eval( msg.decode('utf-8'))
        au_df = au_df.append(aus, ignore_index=True)

print(str(au_df.shape))
print(str(au_df.tail()))
