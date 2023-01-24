import streamlit as st
from streamlit_webrtc import webrtc_streamer
import av
import cv2
import base64
import requests
from io import BytesIO
import matplotlib.image as mpimg
import numpy as np
# import matplotlib.pyplot as plt

flip = st.checkbox("Flip")
url = 'http://13.40.194.22/fmc_api'


def stringToRGB(base64_string):
    imgdata = base64.b64decode(base64_string + '==')
    im = BytesIO(imgdata)
    img = mpimg.imread(im, format='PNG')

    # img = Image.open(im)
    opencv_img= cv2.cvtColor(np.array(img), cv2.COLOR_BGR2RGB)
    return opencv_img 

def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24")
    jpg_img = cv2.imencode('.PNG', img)
    b64_string = base64.b64encode(jpg_img[1]).decode('utf-8')
   
    payload ={ "base64_string": b64_string}
    resp = requests.post(url=url, data=payload) 
    r_json = resp.json()
    img_str = r_json['message']
    im = stringToRGB(img_str)

    # print(im.shape)

    im = cv2.normalize(im, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8U)

    # plt.imshow(im)
    # plt.show()
   
    flipped = img[::-1,:,:] if flip else img

    return av.VideoFrame.from_ndarray(im, format="bgr24")


webrtc_streamer(
    key="example",
    video_frame_callback=video_frame_callback,
    rtc_configuration={  # Add this line
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
    }
)