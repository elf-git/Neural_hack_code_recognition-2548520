import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers, callbacks
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt

# --- 1. Data Preprocessing and Preparation ---
print("Loading MNIST dataset...")
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize pixel values to be between 0 and 1
x_train, x_test = x_train / 255.0, x_test / 255.0

# Reshape data to (28, 28, 1) to specify a single color channel (grayscale)
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)
print(f"x_train shape: {x_train.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"x_test shape: {x_test.shape}")
print(f"y_test shape: {y_test.shape}")

# Convert labels to one-hot encoding
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)

# --- 2. Model Construction ---
print("Building Convolutional Neural Network...")
model = models.Sequential()

# Layer 1: Convolution Layer 1
model.add(layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)))

# Layer 2: Max Pooling Layer 1
model.add(layers.MaxPooling2D(pool_size=(2, 2)))

# Layer 3: Convolution Layer 2
model.add(layers.Conv2D(64, kernel_size=(3, 3), activation='relu'))

# Layer 4: Max Pooling Layer 2
model.add(layers.MaxPooling2D(pool_size=(2, 2)))

# Layer 5: Flatten Layer
model.add(layers.Flatten())

# Layer 6: Fully Connected (Dense Layer)
model.add(layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.001)))

# Layer 7: Dropout Layer
model.add(layers.Dropout(0.5))

# Layer 8: Output Layer
model.add(layers.Dense(10, activation='softmax'))

model.summary()

# --- 3. Model Training ---
print("Compiling model...")
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

print("Starting training...")
batch_size = 128
epochs = 50  # Set to a higher max (dynamically controlled by early stopping)

# Early stopping callback
early_stopping = callbacks.EarlyStopping(
    monitor='val_loss',
    patience=3,
    restore_best_weights=True,
    verbose=1
)

history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    validation_split=0.1,
                    callbacks=[early_stopping])

# --- 4. Validation and Performance Evaluation ---
print("Evaluating model on test data...")
test_loss, test_acc = model.evaluate(x_test,  y_test, verbose=2)
print(f"Test accuracy: {test_acc:.4f}, Test loss: {test_loss:.4f}")

# Save the trained model
model.save("mnist_model.h5")
print("Model saved as 'mnist_model.h5'")

# Plot training & validation accuracy and loss values
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.savefig("training_performance.png")
print("Performance metrics saved as 'training_performance.png'")
