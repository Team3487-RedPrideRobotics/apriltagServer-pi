from math import asin, atan, atan2, sqrt
import math
import cv2
import numpy as np
from dt_apriltags import Detector
from dt_apriltags import Detection

cap = cv2.VideoCapture(0)
at_detector = Detector(searchpath=['apriltags'],
                       families='tag16h5',
                       nthreads=1,
                       quad_decimate=1.0,
                       quad_sigma=1,
                       refine_edges=1,
                       decode_sharpening=0,
                       debug=0)

camera_matrix = [[985.51615849,0.,657.87528672],[0.,984.87908342,354.97674327],[  0.,0.,1.]]
camera_params = [camera_matrix[0][0],camera_matrix[1][1],camera_matrix[0][2],camera_matrix[1][2]]
tag_size = 6*2.54/100 * 1.216# in * cm/in * m/cm
dist_from_obj = 55 * 2.54/100 # in * cm/in * m/cm

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
    yaw = asin(R[2][0]) # from parallel to tag facing direction
    camera_pose = -np.matrix(R).T * np.matrix(det.pose_t) # [[X],[Y],[Z]]
    distance = sqrt(camera_pose[0][0] ** 2 + (camera_pose[2][0]+dist_from_obj) ** 2)
    required_angle = atan(camera_pose[0][0]/(camera_pose[2][0]+dist_from_obj))
    if abs(camera_pose[2][0]) < dist_from_obj:
        required_angle += math.pi/2
    delta_angle = required_angle - yaw

    cv2.putText(frame,str(distance),integerize_tuple(corners[2]),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,0))
    cv2.putText(frame,str(delta_angle),integerize_tuple(corners[0]),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,255))




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
            if(tag.tag_id != 0 or tag.pose_err > 1e-05):
                continue
            
            

            draw_tag(tag,frame)
        cv2.imshow('Frame',frame)
        # Press Q on keyboard to  exit
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
# Break the loop
    else: 
        break


