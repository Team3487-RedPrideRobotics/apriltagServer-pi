#!/usr/bin/python3
from webbrowser import get
from cscore import CameraServer
from networktables import NetworkTables

import cv2
import json
import numpy as np
import time
from detect import get_pose

def main():
    with open('/boot/frc.json') as f:
        config = json.load(f)
    camera = config['cameras'][0]

    width = camera['width']
    height = camera['height']

    if(config['ntmode'] == 'server'):
        NetworkTables.initialize()
    else:
        NetworkTables.startClient('10.34.87.2')


    cs = CameraServer()
    cs.startAutomaticCapture()    
    input_stream = cs.getVideo()
    output_stream = cs.putVideo('Processed', width, height)   
    # Table for vision output information
    vision_nt = NetworkTables.getTable('Vision')   
    # Allocating new images is very expensive, always try to preallocate
    # Wait for NetworkTables to start
    img = np.zeros(shape=(height, width, 3), dtype=np.uint8)
    time.sleep(0.5) 
    while True:
        start_time = time.time()

        frame_time, input_img = input_stream.grabFrame(img)
        vision_nt.putNumber("frame time",frame_time)


        # Notify output of error and skip iteration
        if frame_time == 0:
            output_stream.notifyError(input_stream.getError())
            continue
        output_img,angle,x,y = get_pose(input_img)
        output_stream.putFrame(output_img)
        if(angle):
            vision_nt.putNumber("angle",angle)
            vision_nt.putNumber("x",x*4)
            vision_nt.putNumber("y",y*4)
            NetworkTables.flush()


    
        
      

main()
