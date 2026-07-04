import os
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
from tensorflow.keras import layers
from tensorflow.keras import models

# =========================================================================
# === STEP 1: DEFINE PATHS AND CONSTANTS ==================================
# =========================================================================

# --- Define Paths ---
base_dir = '/Users/srij/Desktop/chest_xray'

train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'val')
test_dir = os.path.join(base_dir, 'test')

# --- Define Constants ---
IMAGE_SIZE = 150
BATCH_SIZE = 32
EPOCHS = 15

# =========================================================================
# === STEP 2: DATA PREPROCESSING AND AUGMENTATION =========================
# =========================================================================
print("--- Preparing Data Generators ---")

train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
)

validation_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary'
)

validation_generator = validation_datagen.flow_from_directory(
    validation_dir,
    target_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary'
)

# =========================================================================
# === STEP 3 (v2): BUILD IMPROVED MODEL WITH DROPOUT ======================
# =========================================================================
from tensorflow.keras.applications import VGG16

# =========================================================================
# === STEP 3 (v3): BUILD MODEL WITH TRANSFER LEARNING (VGG16) ===========
# =========================================================================
print("\n--- Building and Compiling Transfer Learning Model (VGG16) ---")

# 1. Load the VGG16 convolutional base
# We are not including the top (fully-connected) layers
# We specify our input shape
conv_base = VGG16(weights='imagenet',
                  include_top=False,
                  input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3))

# 2. Freeze the convolutional base
# This is the most important step in transfer learning.
# We don't want to ruin the pre-learned features during training.
print("Freezing layers in the VGG16 convolutional base...")
conv_base.trainable = False

# 3. Create our new model and add the VGG16 base
model_v3 = models.Sequential()
model_v3.add(conv_base)

# 4. Add our own new classifier on top
model_v3.add(layers.Flatten())
model_v3.add(layers.Dense(256, activation='relu')) # A dense layer with 256 neurons
model_v3.add(layers.Dropout(0.5)) # Dropout for regularization
model_v3.add(layers.Dense(1, activation='sigmoid')) # Final output layer

# Print the new model's summary
# Notice the huge number of non-trainable parameters!
model_v3.summary()


# 5. Compile the new model
model_v3.compile(optimizer='adam',
                 loss='binary_crossentropy',
                 metrics=['accuracy'])
from tensorflow.keras.applications import VGG16

# =========================================================================
# === STEP 3 (v3): BUILD MODEL WITH TRANSFER LEARNING (VGG16) ===========
# =========================================================================
print("\n--- Building and Compiling Transfer Learning Model (VGG16) ---")

# 1. Load the VGG16 convolutional base
# We are not including the top (fully-connected) layers
# We specify our input shape
conv_base = VGG16(weights='imagenet',
                  include_top=False,
                  input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3))

# 2. Freeze the convolutional base
# This is the most important step in transfer learning.
# We don't want to ruin the pre-learned features during training.
print("Freezing layers in the VGG16 convolutional base...")
conv_base.trainable = False

# 3. Create our new model and add the VGG16 base
model_v3 = models.Sequential()
model_v3.add(conv_base)

# 4. Add our own new classifier on top
model_v3.add(layers.Flatten())
model_v3.add(layers.Dense(256, activation='relu')) # A dense layer with 256 neurons
model_v3.add(layers.Dropout(0.5)) # Dropout for regularization
model_v3.add(layers.Dense(1, activation='sigmoid')) # Final output layer

# Print the new model's summary
# Notice the huge number of non-trainable parameters!
model_v3.summary()


# 5. Compile the new model
model_v3.compile(optimizer='adam',
                 loss='binary_crossentropy',
                 metrics=['accuracy'])

# =========================================================================
# === STEP 4: TRAIN THE IMPROVED MODEL ====================================
# =========================================================================
print("\n--- Starting Model Training ---")

history = model_v3.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=validation_generator
)

print("\n--- Model Training Finished ---")

# =========================================================================
# === STEP 5: EVALUATE AND INTERPRET THE IMPROVED MODEL ===================
# =========================================================================

# --- Plot Learning Curves ---
print("\n--- Generating Learning Curves ---")
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs_range = range(EPOCHS)

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.show()

# --- Final Evaluation on the Test Set ---
print("\n--- Evaluating Model on Test Set ---")
test_generator = validation_datagen.flow_from_directory(
    test_dir,
    target_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=1,
    class_mode='binary',
    shuffle=False
)

test_loss, test_acc = model_v3.evaluate(test_generator)
print(f"Test Accuracy: {test_acc:.4f}")
print(f"Test Loss: {test_loss:.4f}")

# --- Generate Confusion Matrix ---
print("\n--- Generating Confusion Matrix ---")
true_labels = test_generator.classes
pred_probabilities = model_v3.predict(test_generator)
pred_labels = (pred_probabilities > 0.5).astype(int)

cm = confusion_matrix(true_labels, pred_labels)
class_names = ['NORMAL', 'PNEUMONIA']

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names)
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()

# --- Print Classification Report ---
print("\n--- Classification Report ---")
print(classification_report(true_labels, pred_labels, target_names=class_names))
