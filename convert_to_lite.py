import tensorflow as tf

import os
# Convert the model
model_path = 'resnet/best_model.hdf5'
model = tf.keras.models.load_model(model_path)
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
open("resnet/good_model.tflite", "wb").write(tflite_model)