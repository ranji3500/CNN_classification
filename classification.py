import tensorflow as tf
model = tf.keras.models.Sequential()
input_size = (224,224)
model.add(tf.keras.layers.Convolution2D(32,3,3,input_shape = (*input_size,3),activation='relu'))
model.add(tf.keras.layers.MaxPooling2D(pool_size = (2, 2)))
model.add(tf.keras.layers.Convolution2D(32, 3, 3, activation = 'relu'))
model.add(tf.keras.layers.MaxPooling2D(pool_size = (2, 2)))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(units = 64, activation = 'relu'))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(units = 1, activation = 'sigmoid'))
model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
from keras.preprocessing.image import ImageDataGenerator
batch_size = 64
import os
ROOT_DIR = os.path.dirname(os.path.abspath(__file__)) # This is your Project Root

train_path =ROOT_DIR + '/dataset/train'
test_path = ROOT_DIR + '/dataset/test'
# image augmentation part
train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

# create training set
# wanna get higher accuracy -> inccrease target_size
training_set = train_datagen.flow_from_directory(train_path,
                                                 target_size = input_size,
                                                 batch_size = batch_size,
                                                 class_mode = 'binary')

# create test set
# wanna get higher accuracy -> inccrease target_size
test_set = test_datagen.flow_from_directory(test_path,
                                            target_size = input_size,
                                            batch_size = batch_size,
                                            class_mode = 'binary')

# fit the cnn model to the trainig set and testing it on the test set
model.fit(training_set,
          steps_per_epoch = 4662/batch_size,
          epochs = 1,
          validation_data = test_set,
          validation_steps = 1653/batch_size)   #noof_images train and valid
model.save('model.h5')
