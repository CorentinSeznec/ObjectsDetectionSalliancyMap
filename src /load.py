import os 
#from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, array_to_img, load_img
import tensorflow as tf

from model import *
import sys
print(sys.executable)

tf.config.list_physical_devices('GPU')



train_dir = "./train/"
test_dir = "./test/"



train_datagen = tf.keras.preprocessing.image.ImageDataGenerator()
test_datagen = tf.keras.preprocessing.image.ImageDataGenerator()

# Flow training images in batches of 20 using train_datagen generator
train_generator = train_datagen.flow_from_directory(
        train_dir,  # This is the source directory for training images
        target_size=(150, 150),  # All images will be resized to 150x150
        batch_size=45,
        # Since we use binary_crossentropy loss, we need binary labels
        class_mode='categorical')

test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(150, 150),
        batch_size=45,
        class_mode='categorical')



## Create model

base_resnet_net = tf.keras.applications.VGG16(
    weights='imagenet',  # Load weights pre-trained on ImageNet.
    input_shape=(150, 150, 3),
    include_top=False)  # Do not include the ImageNet classifier at the top.

# Freeze the base model
base_resnet_net.trainable = False

# Create a new model on top
inputs = tf.keras.Input(shape=(150, 150, 3))
# We make sure that the base_model is running in inference mode here,
# by passing `training=False`. This is important for fine-tuning, as you will
# learn in a few paragraphs.
x = base_resnet_net(inputs, training=False)
# Convert features of shape `base_model.output_shape[1:]` to vectors
x = tf.keras.layers.GlobalAveragePooling2D()(x)
# A Dense classifier with a single unit (binary classification)
outputs = tf.keras.layers.Dense(9,  activation = 'softmax')(x)
model= tf.keras.Model(inputs, outputs)

#optimizer = tf.keras.optimizers.Adam()

model.compile(optimizer="adam",
                loss="categorical_crossentropy",
                metrics=['accuracy'])

model.summary()


checkpoint_filepath = './checkpoints/checkpoint'
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=True,
    monitor='val_accuracy',
    mode='max',
    save_best_only=True)


history = model.fit(
      train_generator,
      steps_per_epoch=215,  # 2000 images = batch_size * steps
      epochs=10,
      validation_data=test_generator,
      callbacks=[model_checkpoint_callback],
      batch_size=45,
      validation_steps=142,
      verbose=1)


