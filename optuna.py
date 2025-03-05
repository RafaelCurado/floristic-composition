import os
import optuna
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator


os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Disable GPU

# Dataset Configuration
dataset_dir = "FloristicSampleV1"  # Adjust path if needed
img_size = (150, 150)
num_classes = 3  # Arbustos, Gramíneas, Leguminosas

# Function to load data
def load_data(batch_size):
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        validation_split=0.2
    )

    train_generator = train_datagen.flow_from_directory(
        dataset_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode="categorical",
        subset="training"
    )

    val_generator = train_datagen.flow_from_directory(
        dataset_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode="categorical",
        subset="validation"
    )

    return train_generator, val_generator

# Optuna objective function
def objective(trial):
    # Suggest hyperparameters
    filters = trial.suggest_categorical("filters", [32, 64, 128])  # Number of filters
    kernel_size = trial.suggest_categorical("kernel_size", [3, 5])  # Kernel size
    dense_units = trial.suggest_int("dense_units", 64, 256, step=64)  # Dense layer size
    dropout_rate = trial.suggest_float("dropout", 0.2, 0.5)  # Dropout rate
    learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)  # Learning rate
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])  # Batch size

    # Load Data
    train_generator, val_generator = load_data(batch_size)

    # Define Model
    model = models.Sequential([
        layers.Conv2D(filters, (kernel_size, kernel_size), activation="relu", input_shape=(150, 150, 3)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(filters * 2, (kernel_size, kernel_size), activation="relu"),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(dense_units, activation="relu"),
        layers.Dropout(dropout_rate),
        layers.Dense(num_classes, activation="softmax"),
    ])

    # Compile Model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    # Train Model
    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=10,  # Reduce epochs for faster trials
        verbose=0  # Suppress training logs for speed
    )

    # Get Best Validation Accuracy
    val_acc = max(history.history["val_accuracy"])
    
    return val_acc  # Optuna maximizes this



study = optuna.create_study(direction="maximize")  # Maximize accuracy
study.optimize(objective, n_trials=20)  # Run 20 trials

# Print best hyperparameters
print("Best Hyperparameters:", study.best_params)