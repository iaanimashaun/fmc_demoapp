import streamlit as st
from streamlit_webrtc import webrtc_streamer
import av
import cv2
import base64
import requests
from io import BytesIO
import matplotlib.image as mpimg
import numpy as np
from api.ipd import measurements_checks
# import matplotlib.pyplot as plt

flip = st.checkbox("Flip")
url = 'http://13.40.194.22/fmc_api'
url = 'http://13.42.31.235/fmc_api'


def stringToRGB(base64_string):
    imgdata = base64.b64decode(base64_string + '==')
    im = BytesIO(imgdata)
    img = mpimg.imread(im, format='PNG')

    # img = Image.open(im)
    opencv_img= cv2.cvtColor(np.array(img), cv2.COLOR_BGR2RGB)
    return opencv_img 

def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24")

    # jpg_img = cv2.imencode('.PNG', img)
    # b64_string = base64.b64encode(jpg_img[1]).decode('utf-8')
   
    # payload ={ "base64_string": b64_string}
    # resp = requests.post(url=url, data=payload) 
    # r_json = resp.json()
    # img_str = r_json['message']
    # im = stringToRGB(img_str)

    # # print(im.shape)

    # im = cv2.normalize(im, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8U)


    im = measurements_checks(img)

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





# import threading

# import cv2
# import streamlit as st
# from matplotlib import pyplot as plt

# from streamlit_webrtc import webrtc_streamer

# lock = threading.Lock()
# img_container = {"img": None}


# def video_frame_callback(frame):
#     img = frame.to_ndarray(format="bgr24")
#     with lock:
#         img_container["img"] = img

#     return frame


# webrtc_streamer(
#     key="example",
#     video_frame_callback=video_frame_callback,
#     rtc_configuration={  # Add this line
#         "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
#     }
# )



# fig_place = st.empty()
# fig, ax = plt.subplots(1, 1)

# while ctx.state.playing:
#     with lock:
#         img = img_container["img"]
#     if img is None:
#         continue
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     ax.cla()
#     ax.hist(gray.ravel(), 256, [0, 256])
#     fig_place.pyplot(fig)