

import cv2
import numpy as np
import mediapipe as mp
import matplotlib.pyplot as plt
import math

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh

def calc_area(mesh_points, landmarks):
    contour = mesh_points[landmarks]
    area = cv2.contourArea(contour)
    return area


def draw_contour(img, contours):
    cv2.drawContours(img, [contours], -1, (0,255,255), 3)
    return img


def calc_degree(pt1, pt2):
    # Difference in x coordinates
    dx = pt1[0] - pt2[0]

    # Difference in y coordinates
    dy = pt1[1] - pt2[1]

    # Angle between p1 and p2 in radians
    theta = math.atan2(dy, dx)
    
    degree = math.degrees(theta)
    degree = round(degree, 2)

    return degree


def get_mesh_points(image):
        # read image path
    # image = cv2.imread(image_path)

    # get image height and width
    img_h, img_w = image.shape[:2]

    # print(image.shape)

    image_area = img_h * img_w
    # define face mesh object
    face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5)

    # use face mesh object to get facial landmarks
    results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    if results.multi_face_landmarks:
        mesh_points=np.array([np.multiply([p.x, p.y], [img_w, img_h]).astype(int) for p in results.multi_face_landmarks[0].landmark])

        return mesh_points, image_area
    
    return None, image_area



def calc_iris_radius(mesh_points, left_iris, right_iris):
     # estimate the radius and centre of cornea
    (l_cx, l_cy), l_radius = cv2.minEnclosingCircle(mesh_points[left_iris])
    (r_cx, r_cy), r_radius = cv2.minEnclosingCircle(mesh_points[right_iris])
    
    center_left = np.array([l_cx, l_cy], dtype=np.int32)
    center_right = np.array([r_cx, r_cy], dtype=np.int32)

    return center_left, center_right, l_radius, r_radius