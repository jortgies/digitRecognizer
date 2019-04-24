# Larger CNN for the MNIST Dataset
# Convolutional Neural Network
# guide to CNN: https://towardsdatascience.com/a-comprehensive-guide-to-convolutional-neural-networks-the-eli5-way-3bd2b1164a53
# guide to this code: https://machinelearningmastery.com/handwritten-digit-recognition-using-convolutional-neural-networks-python-keras/
# extra: https://gist.github.com/prateekchandrayan/489025f1feef769590bc7b2de1837fca
import numpy as np
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from keras import backend as K

K.set_image_dim_ordering('th')
# fix random seed for reproducibility
seed = 7
np.random.seed(seed)
# load data
(X_train, y_train), (X_test, y_test) = mnist.load_data()
# reshape to be [samples][pixels][width][height]
X_train = X_train.reshape(X_train.shape[0], 1, 28, 28).astype('float32')
X_test = X_test.reshape(X_test.shape[0], 1, 28, 28).astype('float32')
# normalize inputs from 0-255 to 0-1
X_train = X_train / 255
X_test = X_test / 255
# one hot encode outputs
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
num_classes = y_test.shape[1]


# define the larger model
def larger_model():
    # create model
    _model = Sequential()
    _model.add(Conv2D(30, (5, 5), input_shape=(1, 28, 28), activation='relu'))
    _model.add(MaxPooling2D(pool_size=(2, 2)))
    _model.add(Conv2D(15, (3, 3), activation='relu'))
    _model.add(MaxPooling2D(pool_size=(2, 2)))
    _model.add(Dropout(0.2))
    _model.add(Flatten())
    _model.add(Dense(128, activation='relu'))
    _model.add(Dense(50, activation='relu'))
    _model.add(Dense(num_classes, activation='softmax'))
    # Compile model
    _model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return _model


# build the model
model = larger_model()
# Fit the model
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=200)
# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)
print("Large CNN Error: %.2f%%" % (100 - scores[1] * 100))

# Save the model
# serialize model to JSON
model_digit_json = model.to_json()
with open("models/CNN_model.json", "w") as json_file:
    json_file.write(model_digit_json)
# serialize weights to HDF5
model.save('models/CNN_model.h5')
print("Saved model to disk")

