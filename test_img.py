import numpy as np
from keras.preprocessing import image
import  cv2
from keras.models import load_model
from PIL import Image, ImageOps

model = load_model(r"C:\Users\Admin\Desktop\object_detection_gender\gender (1).h5")
img = cv2.imread(r'image.jpg')
face = cv2.resize(img, (224, 224), fx=0, fy=0, interpolation=cv2.INTER_CUBIC)
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
    prediction = 'female'
else:
    prediction = 'male'
