from keras.models import model_from_json
import numpy as np
from PIL import Image

img_width = 28
img_height = 28
# load json and create model
json_file = open('models/CNN_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)

# load weights into new model
loaded_model.load_weights("models/CNN_model.h5")
print("Loaded model from disk")

# make a predictions
# 4D tensor with shape: (batch, channels, rows, cols)
img1 = np.invert(Image.open("userDigits/1.jpeg").convert('L'))
img1 = img1 / 255
img1 = np.expand_dims(img1, axis=0)
img1 = np.expand_dims(img1, axis=1)
pred1 = loaded_model.predict_classes(img1)
prob1 = loaded_model.predict(img1)
print("should predict '1':", pred1, " Props", prob1[0])

img3 = np.invert(Image.open("userDigits/3.jpeg").convert('L'))
img3 = img3/255
img3 = np.expand_dims(img3, axis=0)
img3 = np.expand_dims(img3, axis=1)
pred3 = loaded_model.predict_classes(img3)
prob3 = loaded_model.predict(img3)
print("should predict '3':", pred3, " Props", prob3[0])

img7 = np.invert(Image.open("userDigits/7.jpeg").convert('L'))
img7 = img7/255
img7 = np.expand_dims(img7, axis=0)
img7 = np.expand_dims(img7, axis=1)
pred7 = loaded_model.predict_classes(img7)
prob7 = loaded_model.predict(img7)
print("should predict '7':", pred7, " Props", prob7[0])

img8 = np.invert(Image.open("userDigits/8.jpeg").convert('L'))
img8 = img8/255
img8 = np.expand_dims(img8, axis=0)
img8 = np.expand_dims(img8, axis=1)
pred8 = loaded_model.predict_classes(img8)
prob8 = loaded_model.predict(img8)
print("should predict '8':", pred8, " Props", prob8[0])

imgC = np.invert(Image.open("userDigits/input.jpeg").convert('L'))
imgC = imgC/255
imgC = np.expand_dims(imgC, axis=0)
imgC = np.expand_dims(imgC, axis=1)
predC = loaded_model.predict_classes(imgC)
probC = loaded_model.predict(imgC)
print("should predict what you just have drawn:", predC, " Props", probC[0])


