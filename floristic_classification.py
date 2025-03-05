import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix
import os

# Set TensorFlow logging level (useful for SSH)
tf.get_logger().setLevel('ERROR')

# Ensure matplotlib does not use GUI backend (for SSH)
import matplotlib
matplotlib.use('Agg')

# Confirm GPU availability
print("Available GPUs:", tf.config.list_physical_devices('GPU'))

# Define dataset directory
dataset_dir = 'FloristicSampleV1'  # Change this if needed
img_size = (150, 150)
batch_size = 32
num_classes = 3  # Arbustos, Gramíneas, Leguminosas

# Data Augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2
)

# Load Training Data
train_generator = train_datagen.flow_from_directory(
    dataset_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='training'
)

# Load Validation Data
val_generator = train_datagen.flow_from_directory(
    dataset_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'
)

# Print Class Distribution
class_counts = np.bincount(train_generator.classes)
print("Training class distribution:", class_counts)

# Define Model
model = models.Sequential([
    # First Convolutional Layer
    layers.Conv2D(64, (3, 3), activation='relu', input_shape=(150, 150, 3)),

    # Flatten the output
    layers.Flatten(),
    
    # Output Layer (3 classes)
    layers.Dense(num_classes, activation='softmax')
])

# Compile Model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Print Model Summary
model.summary()

# Train Model
epochs = 20
history = model.fit(
    train_generator,
    steps_per_epoch=len(train_generator),
    validation_data=val_generator,
    validation_steps=len(val_generator),
    epochs=epochs
)

# Save Model Checkpoint
model.save('model_checkpoint.h5')
print("Model saved successfully!")

# Evaluate Model
val_loss, val_accuracy = model.evaluate(val_generator)
print(f"Validation Loss: {val_loss}")
print(f"Validation Accuracy: {val_accuracy}")

# Plot Accuracy & Loss (Saved as Images)
plt.figure()
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Training and Validation Accuracy')
plt.savefig('accuracy_plot.png')  # Save instead of show
print("Saved accuracy plot as accuracy_plot.png")

plt.figure()
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and Validation Loss')
plt.savefig('loss_plot.png')  # Save instead of show
print("Saved loss plot as loss_plot.png")

# Confusion Matrix
predictions = model.predict(val_generator, verbose=1)
y_pred = np.argmax(predictions, axis=1)
y_true = val_generator.classes

cm = confusion_matrix(y_true, y_pred)
class_names = ['Arbustos', 'Gramíneas', 'Leguminosas']

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.savefig('confusion_matrix.png')  # Save instead of show
print("Saved confusion matrix as confusion_matrix.png")
