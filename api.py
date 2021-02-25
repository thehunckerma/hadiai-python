from fastapi import FastAPI, File, UploadFile, Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer
from fastapi.staticfiles import StaticFiles
from PIL import Image
import numpy as np
import uvicorn
import base64
import shutil
import uuid
import cv2
import io
import os

import face_recognition
import json
import requests

if not os.path.exists('images'):
    os.makedirs('images')
if not os.path.exists('images_tmp'):
    os.makedirs('images_tmp')

api = FastAPI()

api.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@api.post("/detect")
async def predict(image: bytes = File(...)):
    return detect_face(image)

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")


@api.post("/recognize/{section_id}/{image_uuid}")
async def recognize(section_id, image_uuid, image: UploadFile = File(...), token: str = Depends(oauth2_scheme)):
    detected = recognizer(image_uuid, image)
    message = ""

    if detected:
        resp = requests.get("http://localhost:8080/api/account",
                            headers={"Authorization": "Bearer " + token})
        if resp.status_code == 200:
            user_id = resp.json()['id']
            if user_id > 0:
                _resp = requests.get(
                    "http://localhost:8080/api/python/presence/" + str(section_id) + "/" + str(user_id))
                if _resp.status_code == 201:
                    return {"success": True, "message": "Successfully marked present"}
                else:
                    message = "Couldn't mark presence"
            else:
                message = "Couldn't get user account ID"
        else:
            message = "Couldn't get user account information"
    else:
        message = "Couldn't recognize user's face"

    return {"success": False, "message": message}

api.mount("/images", StaticFiles(directory="images"), name="images")


@api.post("/upload")
async def image(image: UploadFile = File(...)):
    filename = str(uuid.uuid4()) + '.jpg'
    path = os.path.join('images', filename)
    with open(path, "wb") as buffer:
        shutil.copyfileobj(image.file, buffer)
    open_image = Image.open(path)
    rgb_im = open_image.convert('RGB')
    cropped_image = crop_center_square(rgb_im)
    cropped_image.save(path, quality=95)
    return {"uuid": filename}


def crop_center_square(pil_img):
    img_width, img_height = pil_img.size

    if img_width > img_height:
        crop_size = img_height
    else:
        crop_size = img_width

    return pil_img.crop(((img_width - crop_size) // 2,
                         (img_height - crop_size) // 2,
                         (img_width + crop_size) // 2,
                         (img_height + crop_size) // 2))


def detect_face(binaryimg):
    data = {"success": False}
    if binaryimg is None:
        return data

    # convert the binary image to image
    image = read_cv2_image(binaryimg)

    # convert the image to grayscale
    imagegray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # load the face cascade detector
    facecascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

    # detect faces in the image
    facedetects = facecascade.detectMultiScale(imagegray, scaleFactor=1.1, minNeighbors=5,
                                               minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)

    draw_found_faces(facedetects, image, (0, 255, 0))

    # construct a list of bounding boxes from the detection
    facerect = [(int(fx), int(fy), int(fx + fw), int(fy + fh))
                for (fx, fy, fw, fh) in facedetects]

    retval, buffer = cv2.imencode('.jpg', image)
    # update the data dictionary with the faces detected
    data.update({"num_faces": len(facerect),
                 "faces": facerect,
                 "image": base64.b64encode(buffer),
                 "success": True})

    # return the data dictionary as a JSON response
    return data


def read_cv2_image(binaryimg):
    # return image from buffer
    stream = io.BytesIO(binaryimg)

    image = np.asarray(bytearray(stream.read()), dtype="uint8")
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)

    return image


def draw_found_faces(detected, image, color: tuple):
    for (x, y, width, height) in detected:
        cv2.rectangle(
            image,
            (x, y),
            (x + width, y + height),
            color,
            thickness=2
        )


def recognizer(image_uuid, image):

    unknown_image_path = os.path.join('images_tmp', image_uuid)
    known_image_path = os.path.join("images", image_uuid)

    with open(unknown_image_path, "wb") as buffer:
        shutil.copyfileobj(image.file, buffer)

    known_image = face_recognition.load_image_file(known_image_path)
    known_face_encodings = face_recognition.face_encodings(known_image)

    unknown_image = face_recognition.load_image_file(unknown_image_path)
    unknown_face_encodings = face_recognition.face_encodings(unknown_image)

    detected = False
    if len(unknown_face_encodings) > 0:
        for encoding in unknown_face_encodings:
            if face_recognition.compare_faces(
                    known_face_encodings, encoding, 0):
                detected = True
    return detected


if __name__ == '__main__':
    uvicorn.run(api, host='127.0.0.1', port=8000)
