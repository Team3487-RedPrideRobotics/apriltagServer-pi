import cv2
import json
from math import asin, atan, atan2, sqrt
import math
import cv2
import numpy as np
import apriltag
from gui import draw_bounding_box

# Manual implementation of matrix inversion because np.linalg.inv is slow (source: github.com/SouthwestRoboticsProgramming/TagTracker)
def invert(m):
	m00, m01, m02, m03, m10, m11, m12, m13, m20, m21, m22, m23, m30, m31, m32, m33 = np.ravel(m)
	a2323 = m22 * m33 - m23 * m32
	a1323 = m21 * m33 - m23 * m31
	a1223 = m21 * m32 - m22 * m31
	a0323 = m20 * m33 - m23 * m30
	a0223 = m20 * m32 - m22 * m30
	a0123 = m20 * m31 - m21 * m30
	a2313 = m12 * m33 - m13 * m32
	a1313 = m11 * m33 - m13 * m31
	a1213 = m11 * m32 - m12 * m31
	a2312 = m12 * m23 - m13 * m22
	a1312 = m11 * m23 - m13 * m21
	a1212 = m11 * m22 - m12 * m21
	a0313 = m10 * m33 - m13 * m30
	a0213 = m10 * m32 - m12 * m30
	a0312 = m10 * m23 - m13 * m20
	a0212 = m10 * m22 - m12 * m20
	a0113 = m10 * m31 - m11 * m30
	a0112 = m10 * m21 - m11 * m20

	det = m00 * (m11 * a2323 - m12 * a1323 + m13 * a1223) \
		- m01 * (m10 * a2323 - m12 * a0323 + m13 * a0223) \
		+ m02 * (m10 * a1323 - m11 * a0323 + m13 * a0123) \
		- m03 * (m10 * a1223 - m11 * a0223 + m12 * a0123)
	det = 1 / det

	return np.array([ \
		det *  (m11 * a2323 - m12 * a1323 + m13 * a1223), \
		det * -(m01 * a2323 - m02 * a1323 + m03 * a1223), \
		det *  (m01 * a2313 - m02 * a1313 + m03 * a1213), \
		det * -(m01 * a2312 - m02 * a1312 + m03 * a1212), \
		det * -(m10 * a2323 - m12 * a0323 + m13 * a0223), \
		det *  (m00 * a2323 - m02 * a0323 + m03 * a0223), \
		det * -(m00 * a2313 - m02 * a0313 + m03 * a0213), \
		det *  (m00 * a2312 - m02 * a0312 + m03 * a0212), \
		det *  (m10 * a1323 - m11 * a0323 + m13 * a0123), \
		det * -(m00 * a1323 - m01 * a0323 + m03 * a0123), \
		det *  (m00 * a1313 - m01 * a0313 + m03 * a0113), \
		det * -(m00 * a1312 - m01 * a0312 + m03 * a0112), \
		det * -(m10 * a1223 - m11 * a0223 + m12 * a0123), \
		det *  (m00 * a1223 - m01 * a0223 + m02 * a0123), \
		det * -(m00 * a1213 - m01 * a0213 + m02 * a0113), \
		det *  (m00 * a1212 - m01 * a0212 + m02 * a0112)  \
	]).reshape(4, 4)

class _DetectorOptions: # Converts JSON into object for apriltag's dector to read
    def __init__(self, dict=None):
        if dict:
            for key, value in dict.items():
                setattr(self, key, value)

cap = cv2.VideoCapture(1)



detector = apriltag.Detector(_DetectorOptions(json.load(open('detector.json')))
)
camera_matrix = json.load(open('camera_params.json'))
camera_matrix = (camera_matrix['fx'],camera_matrix['fy'],camera_matrix['cx'],camera_matrix['cy'])
print(camera_matrix)
camera_pose = np.array(json.load(open('cameras.json'))['robot_pose'])
tag_pose = np.array(json.load(open('environment.json'))['tags'][0]['transform'])
tag_size = 0.152 * 1.17

if (cap.isOpened()== False): 
    print("Error opening video stream or file")

# Read until video is completed
while(cap.isOpened()):
# Capture frame-by-frame
    ret, frame = cap.read()
    if ret == True:
        grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        tags = detector.detect(grey)
        for tag in tags:
            pose, e0, e1 = detector.detection_pose(tag, camera_matrix)
            if(tag.tag_id != 0):
                continue
            draw_bounding_box(frame,tag,camera_matrix,pose)
            pose[0][3] *= tag_size
            pose[1][3] *= tag_size
            pose[2][3] *= tag_size
            
            # matrix magic (source: github.com/SouthwestRoboticsProgramming/TagTracker)
            # Find where the camera is if the tag is at the origin
            tag_relative_camera_pose = invert(pose)

			# Find the camera position relative to the tag position
            world_camera_pose = np.matmul(tag_pose, tag_relative_camera_pose)

			# Find the position of the robot from the camera position
            inv_rel_camera_pose = invert(camera_pose)
            robot_pose = np.matmul(world_camera_pose, inv_rel_camera_pose)
            
            print(robot_pose)
            print(math.sqrt((robot_pose[2][3]) ** 2 + (robot_pose[0][3]) ** 2))

        cv2.imshow('Frame',frame)
        # Press Q on keyboard to  exit
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    else: 
        break


