!pip install tensorflow-model-optimization
!pip install kaggle
import tempfile
import os

import tensorflow as tf
import numpy as np

from tensorflow_model_optimization.python.core.keras.compat import keras

%load_ext tensorboard
!wget https://raw.githubusercontent.com/mrdbourke/tensorflow-deep-learning/main/extras/helper_functions.py
from helper_functions import create_tensorboard_callback, plot_loss_curves, unzip_data, compare_historys, walk_through_dir
from google.colab import drive
drive.mount('/content/drive')
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import glob
import shutil
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, Dropout, MaxPool2D, BatchNormalization, Input, Conv2DTranspose, Concatenate
from tensorflow.keras.losses import SparseCategoricalCrossentropy, CategoricalCrossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import cv2
from sklearn.model_selection import train_test_split
import random
import h5py
from IPython.display import display
from PIL import Image as im
import datetime
import random
from tensorflow.keras import layers
#API to fetch dataset
#!kaggle datasets download -d ayush02102001/glaucoma-classification-datasets
# extracting the compessed Dataset
from zipfile import ZipFile
dataset = '/content/main_dataset.zip'

with ZipFile(dataset,'r') as zip:
  zip.extractall()
  print('The dataset is extracted')

train_dir = "/content/training"
test_dir = "/content/testing"
walk_through_dir("/content/testing")
walk_through_dir("/content/training")
import tensorflow as tf
IMG_SIZE = (224,224)
train_data = tf.keras.preprocessing.image_dataset_from_directory(train_dir,
                                                                     label_mode=
                                                                     "categorical",
                                                                     image_size=IMG_SIZE)
test_data = tf.keras.preprocessing.image_dataset_from_directory(test_dir,
                                                                label_mode="categorical",
                                                                image_size=IMG_SIZE,
                                                                shuffle=False)
#-- augumentation

import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.layers import Input, Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.utils import plot_model

# Define the paths
train_dir = '/content/training'
test_dir = '/content/testing'
augmented_train_dir = '/content/augmented_training_all4'
augmented_test_dir = '/content/augmented_testing_all4'

# Ensure the augmented directories exist
os.makedirs(augmented_train_dir, exist_ok=True)
os.makedirs(augmented_test_dir, exist_ok=True)

# Define the data augmentation parameters
data_gen_args = dict(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)
# Create an ImageDataGenerator for augmentation
train_datagen = ImageDataGenerator(**data_gen_args)

def augment_and_save_images(generator, source_dir, target_dir, num_images=1000):
    """Function to augment images and save them to the target directory."""
    for class_name in os.listdir(source_dir):
        class_dir = os.path.join(source_dir, class_name)
        target_class_dir = os.path.join(target_dir, class_name)
        os.makedirs(target_class_dir, exist_ok=True)

        if not os.path.isdir(class_dir):
            continue

        image_files = [os.path.join(class_dir, img) for img in os.listdir(class_dir)
                       if img.lower().endswith(('.jpeg', '.jpg', '.png', '.bmp', '.gif'))]

        # Augment images until we have 1000 images in each class directory
        total_images = len(os.listdir(target_class_dir))
        while total_images < num_images:
            for img_path in image_files:
                img = load_img(img_path)
                x = img_to_array(img)
                x = x.reshape((1,) + x.shape)

                for batch in generator.flow(x, batch_size=1, save_to_dir=target_class_dir,
                                            save_prefix='aug_', save_format='jpeg'):
                    total_images += 1
                    if total_images >= num_images:
                        break

# Augment training images
augment_and_save_images(train_datagen, train_dir, augmented_train_dir)

# Augment testing images (if required, can be skipped for just using original data)
#augment_and_save_images(train_datagen, test_dir, augmented_test_dir)
# Model construction
data_augmentation = Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.1),
    tf.keras.layers.RandomZoom(0.1),
    tf.keras.layers.GaussianNoise(0.1)
], name="data_augmentation")

# Load ResNet50 base model
base_model = tf.keras.applications.EfficientNetB0(
    weights='imagenet',
    include_top=False,
    input_shape=(224,224, 3)
)

base_model.trainable = False

inputs = Input(shape=(224,224, 3), name="input_layer")
x = data_augmentation(inputs)
x = base_model(x)
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
output = Dense(4, activation='softmax')(x)
model = Model(inputs, outputs=output)

# Plot the model
plot_model(model, show_shapes=True, to_file='model_structure.png')

# Summary of the model
model.summary()
import tensorflow as tf

batch_size = 32

# Count the number of samples in the training dataset
total_train_samples = train_data.reduce(0, lambda x, _: x + 1).numpy()
total_test_samples = test_data.reduce(0, lambda x, _: x + 1).numpy()

# Calculate steps_per_epoch and validation_steps
steps_per_epoch = total_train_samples // batch_size
validation_steps = total_test_samples // batch_size

# Define the model and compile it
model.compile(loss="CategoricalCrossentropy",
              optimizer=tf.keras.optimizers.Adam(learning_rate=0.0014125375446227542),
              metrics=["accuracy"])
# Define EarlyStopping callback
early_stopping = EarlyStopping(
    monitor='val_loss',  # Monitor validation loss
    patience=5,          # Stop after 5 epochs with no improvement
    restore_best_weights=True  # Restore the best model weights at the end of training
)
# Fit the model
history1 = model.fit(train_data,
                     epochs=100,
                     steps_per_epoch=steps_per_epoch,
                     validation_data=test_data,
                     validation_steps=validation_steps,
                     callbacks=[early_stopping])
                    #  callbacks=[lr_scheduler])  # Uncomment if you want to use the learning rate scheduler
import numpy as np
import matplotlib.pyplot as plt

# Example learning rates for demonstration
lrs = 1e-4 * (10 ** (np.arange(15)/20))

# Example loss values, replace this with your actual loss values from history
# For demonstration purposes, I'll use a dummy list
loss_values = np.random.random(len(lrs))  # Replace this with `history1.history["loss"]`

# Plotting the graph
plt.figure(figsize=(10, 7))
plt.semilogx(lrs, loss_values)  # we want the x-axis (learning rate) to be log scale
plt.xlabel("Learning Rate")
plt.ylabel("Loss")
plt.title("Learning Rate vs. Loss")

# Finding the index of the minimum loss value
min_loss_index = np.argmin(loss_values)

# Getting the corresponding learning rate
optimal_lr = lrs[min_loss_index]

# Plotting a marker at the lowest point
plt.scatter(optimal_lr, loss_values[min_loss_index], color='red', label='Optimal LR')

plt.legend()
plt.show()

# Printing the optimal learning rate and corresponding loss value
print("Optimal Learning Rate:", optimal_lr)
print("Corresponding Loss Value:", loss_values[min_loss_index])
results_all_fine_tune = model.evaluate(test_data)
results_all_fine_tune
pred_probs = model.predict(test_data, verbose=1)
len(pred_probs)
pred_probs.shape
pred_probs[:10]
# Get the class predicitons of each label
pred_classes = pred_probs.argmax(axis=1)

# How do they look?
pred_classes[:10]
# Note: This might take a minute or so due to unravelling 790 batches
y_labels = []
for images, labels in test_data.unbatch(): # unbatch the test data and get images and labels
  y_labels.append(labels.numpy().argmax()) # append the index which has the largest value (labels are one-hot)
y_labels[:10] # check what they look like (unshuffled)
# Get accuracy score by comparing predicted classes to ground truth labels
from sklearn.metrics import accuracy_score
sklearn_accuracy = accuracy_score(y_labels, pred_classes)
sklearn_accuracy
