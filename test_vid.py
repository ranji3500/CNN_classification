import numpy as np
from keras.preprocessing import image
import  cv2
from keras.models import load_model
from PIL import Image, ImageOps
import os
ROOT_DIR = os.path.dirname(os.path.abspath(__file__)) # This is your Project Root

# video_path =ROOT_DIR + ''
model = load_model(r"video.mp4")
cap = cv2.VideoCapture()
while cap.open():
    _,frame = cap.read()
    face = cv2.resize(frame, (224, 224), fx=0, fy=0, interpolation=cv2.INTER_CUBIC)
    # face = cv2.resize(img, (1400, 1000), interpolation=cv2.INTER_CUBIC)

    # face = cv2.resize(frame, (224, 224))
    im = Image.fromarray(face, 'RGB')
    # Resizing into 128x128 because we trained the model with this image size.
    img_array = np.array(im)
    img_array = img_array / 255
    img_array = np.expand_dims(img_array, axis=0)
    pred = model.predict(img_array)
    print(pred)
    if pred[0][0] == 1:
        prediction = 'class1'
    else:
        prediction = 'crow'

