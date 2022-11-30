import json
from math import asin, atan, atan2, sqrt
import math
import cv2
import numpy as np
from pupil_apriltags import Detector
from pupil_apriltags import Detection

import numpy as np



cap = cv2.VideoCapture(1)
at_detector = Detector(families='tag16h5',
                       nthreads=2,
                       quad_decimate=1.0,
                       quad_sigma=1,
                       refine_edges=1,
                       decode_sharpening=0.25,
                       debug=0)

camera_matrix = json.load(open('matrix.json'))["camera_matrix"]

camera_params = [camera_matrix[0][0],camera_matrix[1][1],camera_matrix[0][2],camera_matrix[1][2]]
print(camera_params)
tag_size = 6 * 130/250# in
dist_from_obj = 0 # in

def integerize_tuple(tup):
    output = []
    for i in tup:
        output.append(int(i))
    return tuple(output)

def draw_tag(det, frame):
    corners = det.corners
    center = det.center
    cv2.circle(frame, integerize_tuple(center), 5, (255,255,255))
    cv2.line(frame,integerize_tuple(corners[0]),integerize_tuple(corners[1]),(255,0,0),3)
    cv2.line(frame,integerize_tuple(corners[1]),integerize_tuple(corners[2]),(0,255,0),3)
    cv2.line(frame,integerize_tuple(corners[2]),integerize_tuple(corners[3]),(0,0,255),3)
    cv2.line(frame,integerize_tuple(corners[3]),integerize_tuple(corners[0]),(255,0,255),3)

    R = det.pose_R.transpose()
    t = det.pose_t
    P = np.array([[1,0,0],[0,-1,0],[0,0,-1]])
    yaw = asin(R[2][0]) # from parallel to tag facing direction
    tag_x, _, tag_z = -R * np.matrix(det.pose_t)
    tag_z = float(-tag_z)
    tag_x = float(tag_x)

    required_angle = atan(tag_x/(tag_z+dist_from_obj))
    print(required_angle)

    #print(list(det.pose_R[0]).append(det.pose_t[0][0]))
    #print(tag_size*camera_matrix[0][0]/math.sqrt((corners[0][0]-corners[1][0])**2 + (corners[0][1]-corners[1][1])**2))
    cv2.putText(frame,str(det.pose_t[1]),integerize_tuple(corners[2]),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,0))
    cv2.putText(frame,str(180/math.pi * yaw),integerize_tuple(corners[0]),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,255))




# Check if camera opened successfully
if (cap.isOpened()== False): 
    print("Error opening video stream or file")

# Read until video is completed
while(cap.isOpened()):
# Capture frame-by-frame
    ret, frame = cap.read()
    if ret == True:
        # Display the resulting frame
        grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        tags = at_detector.detect(grey, estimate_tag_pose=True, camera_params=camera_params, tag_size=tag_size)
        for tag in tags:
            if(tag.tag_id != 0 or tag.pose_err > 0.1):
                continue
            
            

            draw_tag(tag,frame)
        cv2.imshow('Frame',frame)
        # Press Q on keyboard to  exit
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
# Break the loop
    else: 
        break


