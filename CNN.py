import tensorflow as tf
from tensorflow.keras.models import model_from_json
from tensorflow.keras.utils import to_categorical
import matplotlib as plt
from tensorflow.keras.datasets import mnist
from keras.initializers.initializers_v1 import RandomNormal
from keras.layers import Dense, BatchNormalization, Dropout, Flatten, MaxPooling2D, Conv2D
from keras import Sequential
import numpy as np

(x_train, y_train), (x_test, y_test) = mnist.load_data()

def data_prep(x_train, x_test, y_train, y_test):
    x_train = x_train.reshape(60000, 28, 28, 1)
    x_test = x_test.reshape(10000, 28, 28, 1)

    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)

    return x_train, x_test, y_train, y_test


x_train, x_test, y_train, y_test = data_prep(x_train, x_test, y_train, y_test)

# CNN modeli

model = Sequential()

model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1), padding='same',  kernel_initializer='glorot_uniform', bias_initializer=RandomNormal(stddev=0.01)))
model.add(BatchNormalization())

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.20))

model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.20))

model.add(Conv2D(256, (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.20))

model.add(Conv2D(512, (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.20))

model.add(Flatten())
model.add(Dense(10, activation='softmax')) #10 sınıflı bir çıkış katmanı

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

with tf.device('/GPU:0'):
    model_train = model.fit(x_train, y_train, batch_size=32, epochs=5, validation_data=(x_test, y_test))

model_json = model.to_json()
open('CNN_six_layer_fully_connected.json', 'w').write(model_json)
model.save_weights('CNN_six_fully_connected.h5', overwrite=True)
print(model_train.history.keys())

plt.plot(model_train.history['accuracy'], label='Training Accuracy')
plt.plot(model_train.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.plot(model_train.history['loss'], label='Training Loss')
plt.plot(model_train.history['val_loss'], label='Validation Loss')
plt.ylabel('Loss')
plt.ylabel('Accuracy')
print("Model Summary =>", model.summary())
plt.legend()
plt.show()

def load_model_and_predict(image_number):
    labels = '''sıfır bir iki üç dört beş altı yedi sekiz dokuz'''.split()

    json_file = open('CNN_six_layer_fully_connected.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)

    loaded_model.load_weights('CNN_six_fully_connected.h5')

    loaded_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    (_, _), (x_test, y_test) = mnist.load_data()

    x_test = x_test.reshape(60000, 28, 28, 1)
    y_test = to_categorical(y_test)

    image_data = x_test[image_number]

    reshaped_input = np.expand_dims(image_data, axis=0)

    predictions = loaded_model.predict(reshaped_input)

    predicted_class = np.argmax(predictions)

    original_label = labels[np.argmax(y_test[image_number])]

    plt.imshow(image_data)
    plt.show()

    print("Original label is {} and predicted label is {}".format(original_label, labels[predicted_class]))
for i in range(0, 20):
    load_model_and_predict(i, 255.0, 10)