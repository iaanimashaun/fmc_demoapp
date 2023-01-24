

import cv2
import numpy as np
import mediapipe as mp
import matplotlib.pyplot as plt
import math
from .utils import *


mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh


landmarks = {
    'silhouette': [
    10,  338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
    397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
    172, 58,  132, 93,  234, 127, 162, 21,  54,  103, 67,  109
  ],


  'right_silhoutte': [10,  338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
    397, 365, 379, 378, 400, 377, 152],

  'left_silhoutte': [152, 148, 176, 149, 150, 136,
    172, 58,  132, 93,  234, 127, 162, 21,  54,  103, 67,  109, 10],


  'lipsUpperOuter': [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291],
  'lipsLowerOuter': [146, 91, 181, 84, 17, 314, 405, 321, 375, 291],
  'lipsUpperInner': [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308],
  'lipsLowerInner': [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308],

  'rightEyeUpper0': [246, 161, 160, 159, 158, 157, 173],
  'rightEyeLower0': [33, 7, 163, 144, 145, 153, 154, 155, 133],
  'rightEyeUpper1': [247, 30, 29, 27, 28, 56, 190],
  'rightEyeLower1': [130, 25, 110, 24, 23, 22, 26, 112, 243],
  'rightEyeUpper2': [113, 225, 224, 223, 222, 221, 189],
  'rightEyeLower2': [226, 31, 228, 229, 230, 231, 232, 233, 244],
  'rightEyeLower3': [143, 111, 117, 118, 119, 120, 121, 128, 245],

  'rightEyebrowUpper': [156, 70, 63, 105, 66, 107, 55, 193],
  'rightEyebrowLower': [35, 124, 46, 53, 52, 65],

  'rightEyeIris': [473, 474, 475, 476, 477],

  'leftEyeUpper0': [466, 388, 387, 386, 385, 384, 398],
  'leftEyeLower0': [263, 249, 390, 373, 374, 380, 381, 382, 362],
  'leftEyeUpper1': [467, 260, 259, 257, 258, 286, 414],
  'leftEyeLower1': [359, 255, 339, 254, 253, 252, 256, 341, 463],
  'leftEyeUpper2': [342, 445, 444, 443, 442, 441, 413],
  'leftEyeLower2': [446, 261, 448, 449, 450, 451, 452, 453, 464],
  'leftEyeLower3': [372, 340, 346, 347, 348, 349, 350, 357, 465],

  'leftEyebrowUpper': [383, 300, 293, 334, 296, 336, 285, 417],
  'leftEyebrowLower': [265, 353, 276, 283, 282, 295],

  'leftEyeIris': [468, 469, 470, 471, 472],

  'midwayBetweenEyes': [168],

  'noseTip': [1],
  'noseBottom': [2],
  'noseRightCorner': [98],
  'noseLeftCorner': [327],

  'rightCheek': [205],
  'leftCheek': [425]
}

def plt_imshow(image, title='image'):
  # convert the image frame BGR to RGB color space and display it
	image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	plt.imshow(image)
	plt.title(title)
	plt.grid(False)
	plt.show()



# @app.route("/blend", methods=["POST"])
def measurements_checks(image):

    # image = cv2.imread(image_path)
    # filename1 = request.form['image']

    mesh_points, image_area = get_mesh_points(image)

    if mesh_points is None:
      return image
    # get indices of landmarks we want to extract (both iris and nasal bridge)
    left_iris, right_iris = landmarks['rightEyeIris'], landmarks['leftEyeIris']
    silhoutte = landmarks['silhouette']
    right_silhoutte = landmarks['right_silhoutte']
    left_silhoutte = landmarks['left_silhoutte']
    # midwayBetweenEyes = landmarks['midwayBetweenEyes']
    top = mesh_points[10]
    bottom = mesh_points[152]

    nose_tip = mesh_points[landmarks['noseTip']][0]
    right_cheek = mesh_points[landmarks['rightCheek']][0]
    left_cheek = mesh_points[landmarks['leftCheek']][0]

    center_left, center_right, l_radius, r_radius = calc_iris_radius(mesh_points, left_iris, right_iris)


    img = image.copy()
    
    _ = cv2.line(img, nose_tip, right_cheek, (0, 0, 255), 2)
    # _ = cv2.line(img, nose_tip, left_cheek, (0, 0, 255), 2)
    _ = cv2.line(img, top, bottom, (0, 0, 255), 2)
    _ = cv2.line(img, center_left, center_right, (0, 0, 255), 2)


    transverse_angle = calc_degree(nose_tip, right_cheek)
    

    horizontal_angle = calc_degree(center_left, center_right)
    vertical_angle = calc_degree(top, bottom)


    
    right_silhoutte_area = calc_area(mesh_points, right_silhoutte)

    left_silhoutte_area = calc_area(mesh_points, left_silhoutte)

    face_area = calc_area(mesh_points, silhoutte)
    facial_ratio = round((right_silhoutte_area / left_silhoutte_area), 2)
    face_image_ratio = round((face_area / image_area), 2)

    # img = image.copy()

    # draw points and lines on image
    cv2.circle(img, center_left, int(l_radius), (255,0,255), 1, cv2.LINE_AA)
    cv2.circle(img, center_right, int(r_radius), (255,0,255), 1, cv2.LINE_AA)

    # cv2.line(img, nose_bridge, center_right, (0, 0, 255), 2)
    # cv2.line(img, nose_bridge, center_left, (0, 255, 0), 2)

    img = cv2.putText(img, f'Right/Left Ratio: {str(facial_ratio)}', (50,30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)
    img = cv2.putText(img, f'Vertical angle: {str(vertical_angle)}', (50,50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
    img = cv2.putText(img, f'Horizontal angle: {str(horizontal_angle)}', (50,70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 2)
    img = cv2.putText(img, f'Face area: {str(face_area)}', (50,90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 2)
    img = cv2.putText(img, f'Image area: {str(image_area)}', (50,110), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 2)
    img = cv2.putText(img, f'face/Image Ratio: {str(face_image_ratio)}', (50,130), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 2)
    img = cv2.putText(img, f'transverse angle: {str(transverse_angle)}', (50,150), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 2)
    # img = cv2.putText(img, f'nose to left cheek angle: {str(ang2)}', (50,170), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 2)

    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	
    # display image
    # plt.imshow(img)
    # plt.title('image')
    # plt.grid(False)
    # plt.show()

    return img







