from keras.models import model_from_json
import numpy as np
from PIL import Image

img_width = 28
img_height = 28
# load json and create model
json_file = open('models/MLP_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)

# load weights into new model
loaded_model.load_weights("models/MLP_model.h5")
print("Loaded model from disk")

# make a prediction
img1 = np.invert(Image.open("userDigits/1.jpeg").convert('L')).ravel()
img1 = img1/255
pred1 = loaded_model.predict_classes(np.array([img1]))
prob1 = loaded_model.predict(np.array([img1]))
print("should predict '1':", pred1, " Prob", prob1)

img3 = np.invert(Image.open("userDigits/3.jpeg").convert('L')).ravel()
img3 = img3/255
pred3 = loaded_model.predict_classes(np.array([img3]))
prob3 = loaded_model.predict(np.array([img3]))
print("should predict '3':", pred3, " Prob", prob3)

img7 = np.invert(Image.open("userDigits/7.jpeg").convert('L')).ravel()
img7 = img7/255
pred7 = loaded_model.predict_classes(np.array([img7]))
prob7 = loaded_model.predict(np.array([img7]))
print("should predict '7':", pred7, " Prob", prob7)

img8 = np.invert(Image.open("userDigits/8.jpeg").convert('L')).ravel()
img8 = img8/255
pred8 = loaded_model.predict_classes(np.array([img8]))
prob8 = loaded_model.predict(np.array([img8]))
print("should predict '8':", pred8, " Prob", prob8)

