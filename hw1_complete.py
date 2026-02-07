#!/usr/bin/env python

import tensorflow as tf
import keras
from keras import layers
import numpy as np
##import ssl
import os

# Fix SSL certificate verification issue on macOS
##ssl._create_default_https_context = ssl._create_unverified_context

def build_model1():
    model = keras.Sequential([
        layers.Flatten(input_shape=(32, 32, 3)),
        layers.Dense(128, activation='leaky_relu'), # Added activation
        layers.Dense(128, activation='leaky_relu'), # Added activation
        layers.Dense(128, activation='leaky_relu'), # Added activation
        layers.Dense(10)
    ])
    model.compile(
        optimizer='adam',
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )
    return model

def build_model2():
    model = keras.Sequential([
        layers.Input(shape=(32, 32, 3)),

        layers.Conv2D(32, (3, 3), strides=(2, 2), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), strides=(2, 2), padding='same', activation='relu'),
        layers.BatchNormalization(),

        layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.Flatten(),
        layers.Dense(10)
    ])
    model.compile(
        optimizer='adam',
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )
    return model

def build_model3():
    model = keras.Sequential([
        layers.Input(shape=(32, 32, 3)),

        layers.SeparableConv2D(32, (3, 3), strides=(2, 2), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.SeparableConv2D(64, (3, 3), strides=(2, 2), padding='same', activation='relu'),
        layers.BatchNormalization(),

        layers.SeparableConv2D(128, (3, 3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.SeparableConv2D(128, (3, 3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.SeparableConv2D(128, (3, 3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.SeparableConv2D(128, (3, 3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.Flatten(),
        layers.Dense(10)
    ])
    model.compile(
        optimizer='adam',
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )
    return model 

def build_model50k():
    model = keras.Sequential([
        layers.Input(shape=(32, 32, 3)),
        layers.Conv2D(16, (3, 3), strides=2, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.SeparableConv2D(32, (3, 3), strides=2, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.SeparableConv2D(64, (3, 3), strides=2, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.SeparableConv2D(64, (3, 3), strides=2, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.GlobalAveragePooling2D(),
        layers.Dense(10)
    ])
   # Dynamic pathing for GitHub Autograder compatibility
    base_path = os.path.dirname(__file__)
    weights_path = os.path.join(base_path, "best_model.h5")

    if os.path.exists(weights_path):
        print(f"DEBUG: Loading weights from {weights_path}")
        model.load_weights(weights_path)
    else:
        print(f"DEBUG: {weights_path} NOT FOUND")

    model.compile(
        optimizer='adam',
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )
    return model

if __name__ == '__main__':
    # Data Loading
    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.cifar10.load_data()
    train_images, test_images = train_images / 255.0, test_images / 255.0
    val_images, val_labels = train_images[:5000], train_labels[:5000]
    train_images, train_labels = train_images[5000:], train_labels[5000:]

  # model1 = build_model1()
    # model1.fit(train_images, train_labels, epochs=30, validation_data=(val_images, val_labels))
    
    # model2 = build_model2()
    # model2.fit(train_images, train_labels, epochs=30, validation_data=(val_images, val_labels))
    
    # model3 = build_model3()
    # model3.fit(train_images, train_labels, epochs=30, validation_data=(val_images, val_labels))
    
    best_model = build_model50k()
    ##print(best_model.summary())
    best_model.fit(train_images, train_labels, epochs=10, validation_data=(val_images, val_labels))
    best_model.save("best_model.h5")

    
    



  
  

