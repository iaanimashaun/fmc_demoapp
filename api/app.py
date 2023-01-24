

import sys
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from PIL import Image
from .ipd import measurements_checks
import cv2
import numpy as np
from pydantic import BaseModel
import base64
from starlette.responses import StreamingResponse
from io import BytesIO
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Response
from PIL import Image
import uvicorn
from mangum import Mangum


app = FastAPI()
# handler = Mangum(app)



@app.get("/")
async def root():
    return {"message": "Welcome to my face measurement api!"}


def stringToRGB(base64_string):
    imgdata = base64.b64decode(str(base64_string))
    im = BytesIO(imgdata)
    # img = mpimg.imread(im, format='JPG')

    img = Image.open(im)
    opencv_img= cv2.cvtColor(np.array(img), cv2.COLOR_BGR2RGB)
    return opencv_img 


@app.post("/")
def fmc_api( base64_string: str = Form(...)):
    # image_as_bytes = str.encode(base64_string)  # convert string to bytes
    img = stringToRGB(base64_string)

    processed_image = measurements_checks(img)

    _, encoded_img = cv2.imencode('.PNG', processed_image)

    encoded_img = base64.b64encode(encoded_img).decode('utf-8')
    
    print(type(encoded_img))
    return {"message": encoded_img} 


# if __name__ == "__main__":
#     app.run(debug=True)
handler = Mangum(app)


if __name__ == "__main__":
    uvicorn.run("app", host="0.0.0.0", port=5000, reload=True)














