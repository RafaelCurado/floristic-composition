import tensorflow as tf

# Load the trained model
model = tf.keras.models.load_model("../basic_cnn_plant_classifier.h5")

# Convert it to TensorFlow Lite format
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the model
with open("model.tflite", "wb") as f:
    f.write(tflite_model)
