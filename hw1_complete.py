#!/usr/bin/env python

# TensorFlow and tf.keras
import tensorflow as tf
import keras
from keras import Input, layers, Sequential

# Helper libraries
import argparse
import numpy as np
# import matplotlib.pyplot as plt  # Keep plotting commented
from matplotlib import image
import ssl
import os

# Fix SSL certificate verification issue on macOS
# This disables SSL verification for dataset downloads (acceptable for development)
ssl._create_default_https_context = ssl._create_unverified_context

print(f"TensorFlow Version: {tf.__version__}")
print(f"Keras Version: {keras.__version__}")


## 

def build_model1():
  model = keras.Sequential([
    layers.Flatten(input_shape = (32, 32, 3)),
    
    #first dense layer
    layers.Dense(128),

    ##second dense layer
    layers.Dense(128),

    ##third dense layer
    layers.Dense(128),

    ##fourth dense layer
    layers.Dense(10)


  ]) # Add code to define model 1.
  model.compile(
    optimizer = 'adam',
    loss = keras.losses.SparseCategoricalCrossentropy(from_logits = True),
    metrics = ['accuracy']
  )
  return model

def build_model2():
    model = keras.Sequential([
        layers.Conv2D(32, (3,3), strides = (2,2), padding = 'same', activation = 'relu', input_shape = (32, 32, 3)),
        layers.BatchNormalization(),

        layers.Conv2D(64, (3,3), strides = (2,2), padding = 'same', activation = 'relu'),
        layers.BatchNormalization(),

        layers.Conv2D(128, (3,3), strides = (1,1), padding = 'same', activation = 'relu'),
        layers.BatchNormalization(),

        layers.Conv2D(128, (3,3), strides = (1,1), padding = 'same', activation = 'relu'),
        layers.BatchNormalization(),

        layers.Conv2D(128, (3,3), strides = (1,1), padding = 'same', activation = 'relu'),
        layers.BatchNormalization(),

        layers.Conv2D(128, (3,3), strides = (1,1), padding = 'same', activation = 'relu'),
        layers.BatchNormalization(),

        layers.Flatten(),
        layers.Dense(10)
    ])
    return model

def build_model3():
  model = keras.Sequential([
    layers.Input(shape = (32, 32, 3)), # Add code to define model 3.
    layers.SeparableConv2D(32, (3,3), strides = 2, padding = 'same', activation = 'relu'),
    layers.BatchNormalization(),
    
    layers.SeparableConv2D(64, (3,3), strides = 2, padding = 'same', activation = 'relu'),
    layers.BatchNormalization(),
    
    layers.SeparableConv2D(128, (3,3), padding = 'same', activation = 'relu'),
    layers.BatchNormalization(),

    layers.SeparableConv2D(128, (3,3), padding = 'same', activation = 'relu'),
    layers.BatchNormalization(),

    layers.SeparableConv2D(128, (3,3), padding = 'same', activation = 'relu'),
    layers.BatchNormalization(),

    layers.SeparableConv2D(128, (3,3), padding = 'same', activation = 'relu'),
    layers.BatchNormalization(),

    layers.Flatten(),
    layers.Dense(10)
  ])
  model.compile(
    optimizer = 'adam',
    loss = keras.losses.SparseCategoricalCrossentropy(from_logits = True),
    metrics = ['accuracy']
  )
  return model 
  ## This one should use the functional API so you can create the residual connections

def build_model50k():
    model = keras.Sequential([
        layers.Input(shape = (32, 32, 3)),
        
        layers.Conv2D(16, (3,3), strides = 2, padding = 'same', activation = 'relu'),
        layers.BatchNormalization(),
        
        layers.SeparableConv2D(32, (3,3), strides = 2, padding = 'same', activation = 'relu'),
        layers.BatchNormalization(),
        
        layers.SeparableConv2D(64, (3,3), strides = 2, padding = 'same', activation = 'relu'),
        layers.BatchNormalization(),

        layers.SeparableConv2D(64, (3,3), strides = 2, padding = 'same', activation = 'relu'),
        layers.BatchNormalization(),

        layers.GlobalAveragePooling2D(),
        layers.Dense(10)
    ])
    
    model.compile(
        optimizer = 'adam',
        loss = keras.losses.SparseCategoricalCrossentropy(from_logits = True),
        metrics = ['accuracy']
    )
    return model

# no training or dataset construction should happen above this line
# also, be careful not to unindent below here, or the code be executed on import
if __name__ == '__main__':

  ########################################
  ## Add code here to Load the CIFAR10 data set
  (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.cifar10.load_data()

  train_images = train_images / 255.0
  test_images  = test_images  / 255.0

  val_images = train_images[:5000]
  val_labels = train_labels[:5000]
  train_images = train_images[5000:]
  train_labels = train_labels[5000:]

  ########################################
  ## Build and train model 1
# model1 = build_model1()
# ##summary method 
# model1.summary()
# #   # compile and train model 1.
# model1.fit(train_images, train_labels, epochs = 30, validation_data = (val_images, val_labels))
# model1.evaluate(test_images, test_labels)


  ## Build, compile, and train model 2 (DS Convolutions)
  model2 = build_model2()
  model2.compile(
      optimizer='adam',
      loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
      metrics=['accuracy']
  )
  model2.summary()

  history = model2.fit(train_images, train_labels, epochs = 30, validation_data = (val_images, val_labels))

  # Get final accuracies
  final_train_acc = history.history['accuracy'][-1]
  final_val_acc = history.history['val_accuracy'][-1]
  test_loss, test_acc = model2.evaluate(test_images, test_labels, verbose=0)

  # Print accuracies
  print("\n" + "="*50)
  print("FINAL ACCURACIES:")
  print("="*50)
  print(f"Training Accuracy:   {final_train_acc:.4f} ({final_train_acc*100:.2f}%)")
  print(f"Validation Accuracy: {final_val_acc:.4f} ({final_val_acc*100:.2f}%)")
  print(f"Test Accuracy:       {test_acc:.4f} ({test_acc*100:.2f}%)")
  print("="*50 + "\n")

  # Keep plotting code commented
  # plt.plot(history.history['accuracy'], label='accuracy')
  # plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
  # plt.xlabel('Epoch')
  # plt.ylabel('Accuracy')
  # plt.ylim([0, 1])
  # plt.legend(loc='lower right')
  # plt.show()

  ## image classification``
  class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

  # Check if image file exists
  image_path = './rsz_car_o.png'
  if not os.path.exists(image_path):
      # Try alternative paths
      alt_paths = ['rsz_car_o.png', 'rsz_car_o.PNG', './rsz_car_o.PNG']
      found = False
      for alt_path in alt_paths:
          if os.path.exists(alt_path):
              image_path = alt_path
              found = True
              break
      if not found:
          print(f"Warning: Image file not found. Skipping image classification.")
          image_path = None

  if image_path and os.path.exists(image_path):
      test_img = np.array(tf.keras.utils.load_img(
        image_path,
        target_size = (32, 32),
        color_mode = 'rgb',
        
      ))

      test_img_input = np.expand_dims(test_img / 255.0, axis = 0)

      ## Run prediction
      raw_predictions = model2.predict(test_img_input)
      predicted_index = np.argmax(raw_predictions)

      print(f"The model thinks this is a {class_names[predicted_index]}")

  ## Build, compile, and train model 3
  model3 = build_model3()
  model3.compile(
      optimizer='adam',
      loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
      metrics=['accuracy']
  )
  model3.summary()

  history3 = model3.fit(train_images, train_labels, epochs = 30, validation_data = (val_images, val_labels))

  print("\nModel 3 Final Test Evaluation")
  test_loss, test_acc = model3.evaluate(test_images, test_labels)

  final_train_acc = history3.history['accuracy'][-1]
  final_val_acc = history3.history['val_accuracy'][-1]
  final_test_acc, final_test_loss = model3.evaluate(test_images, test_labels, verbose=0)

  print("\n" + "="*50)
  print("FINAL ACCURACIES:")
  print("="*50)
  print(f"Training Accuracy:   {final_train_acc:.4f} ({final_train_acc*100:.2f}%)")
  print(f"Validation Accuracy: {final_val_acc:.4f} ({final_val_acc*100:.2f}%)")
  print(f"Test Accuracy:       {final_test_acc:.4f} ({final_test_acc*100:.2f}%)")
  print(f"Test Loss:           {final_test_loss:.4f}")
  print("="*50 + "\n")

  ### Build and display summary for the 50k model
  best_model = build_model50k()
  best_model.summary()

  best_model.fit(train_images, train_labels, epochs=50, 
                 validation_data=(val_images, val_labels))

  # Save the file for submission
  best_model.save("best_model.h5")



  
  

