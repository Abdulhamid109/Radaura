from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse
import numpy as np
import cv2
import io

app = FastAPI()

# Load model files
prototxt = 'models\colorization_deploy_v2.prototxt'
points = 'models\pts_in_hull.npy'
model = 'models\colorization_release_v2.caffemodel'

net = cv2.dnn.readNetFromCaffe(prototxt=prototxt, caffeModel=model)
pts = np.load(points)
pts = pts.transpose().reshape(2, 313, 1, 1)

class8 = net.getLayerId("class8_ab")
conv8 = net.getLayerId("conv8_313_rh")
net.getLayer(class8).blobs = [pts.astype("float32")]
net.getLayer(conv8).blobs = [np.full([1, 313], 2.606, dtype="float32")]



@app.get('/')
def starter():
    return {
        'success':'Backend is running properly'
    }

@app.post("/colorize/")
async def colorize_image(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if image is None:
        return {"error": "Invalid image file"}

    scaled = image.astype("float32") / 255.0
    lab = cv2.cvtColor(scaled, cv2.COLOR_BGR2LAB)

    resized = cv2.resize(lab, (224, 224))
    L = cv2.split(resized)[0]
    L -= 50

    net.setInput(cv2.dnn.blobFromImage(L))
    ab = net.forward()[0, :, :, :].transpose((1, 2, 0))
    ab = cv2.resize(ab, (image.shape[1], image.shape[0]))

    L = cv2.split(lab)[0]
    colorized = np.concatenate((L[:, :, np.newaxis], ab), axis=2)
    colorized = cv2.cvtColor(colorized, cv2.COLOR_LAB2BGR)
    colorized = np.clip(colorized, 0, 1)
    colorized = (255 * colorized).astype("uint8")

    _, buffer = cv2.imencode(".jpg", colorized)
    return StreamingResponse(io.BytesIO(buffer.tobytes()), media_type="image/jpeg")
