#TensorFLow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

print(tf.__version__)

#import data
fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

#class names, not store with dataset so need to be stored to use later when plotting images
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

#test show how many images and the shape of data
print(train_images.shape)

#test show length of training labels
print(len(train_labels))

#test what is train_labels ... array of integers between 0 and 9
print(train_labels)

#test what is test images
print(test_images.shape)
print(len(test_labels))

#test inspecting the first image
plt.figure()
plt.imshow(train_images[0])
plt.colorbar()
plt.grid(False)
# plt.show()

#preprocess the data/ training the data
#scale values 0 to 1 instead of 0 to 255
train_images = train_images / 255.0
test_images = test_images / 255.0

#display the first 25 images from training set and display class name below each iamge
plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5, 5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])
# plt.show()

#build machine learning model
#setup the layers 
#layers extract representations from data input

model = keras.Sequential([
    #first layer transforms format of images from 2d array to 1d array
    keras.layers.Flatten(input_shape=(28, 28)),
    #first densely-connected (fully-connected) neural layers with 128 nodes
    keras.layers.Dense(128, activation=tf.nn.relu),
    #second dense layer with 10 nodes softmax layer-returns array of 10 probability scores that sum to 1; each node contains score that shows probability image belongs to this class
    keras.layers.Dense(10, activation=tf.nn.softmax)
])

#compile the model
#add settings before ready for training
#loss function-measures how accurate, minimize loss function to steer the model
#optimizer-how model is updated based on data it sees and its loss function
#metrics-used to monitor the training and testing steps; this example uses accuracy-the fraction of images correctly classified
#study/list out different settings and models and pros and cons
model.compile(optimizer=tf.train.AdamOptimizer(), 
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

#training the model
#1. feed training data to model, train_)images and train_labels
#2. model learns to associate images and labels
#3. ask model to make predictions, verify that predictions match the labels
model.fit(train_images, train_labels, epochs=5)

#evaluate accuracy
test_loss, test_acc = model.evaluate(test_images, test_labels)
print('Test accuracy:', test_acc)

#overfitting = when model performs worse on new data than training data

#making predictions
predictions = model.predict(test_images)
print(predictions[0])
#prediction is an array of ten numbers that describe confidence of image corresponds to each of ten classes
#label with highest confidence value
print(np.argmax(predictions[0]))
#check test label to see if correct
print(test_labels[0])

#graph to look at all ten channels; UNDERSTAND GRAPHS
def plot_image(i, predictions_array, true_label, img):
    predictions_array, true_label, img = predictions_array[i], true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img, cmap=plt.cm.binary)

    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'
    plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                100*np.max(predictions_array),
                                class_names[true_label]),
                                color=color)

def plot_value_array(i, predictions_array, true_label):
  predictions_array, true_label = predictions_array[i], true_label[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])
  thisplot = plt.bar(range(10), predictions_array, color="#777777")
  plt.ylim([0, 1]) 
  predicted_label = np.argmax(predictions_array)
 
  thisplot[predicted_label].set_color('red')
  thisplot[true_label].set_color('blue')

i = 0
plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plot_image(i, predictions, test_labels, test_images)
plt.subplot(1,2,2)
plot_value_array(i, predictions,  test_labels)


i = 12
plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plot_image(i, predictions, test_labels, test_images)
plt.subplot(1,2,2)
plot_value_array(i, predictions,  test_labels)

# Plot the first X test images, their predicted label, and the true label
# Color correct predictions in blue, incorrect predictions in red
num_rows = 5
num_cols = 3
num_images = num_rows*num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images):
  plt.subplot(num_rows, 2*num_cols, 2*i+1)
  plot_image(i, predictions, test_labels, test_images)
  plt.subplot(num_rows, 2*num_cols, 2*i+2)
  plot_value_array(i, predictions, test_labels)

# Grab an image from the test dataset
img = test_images[0]
print(img.shape)

#tf.keras models are optimzied to make predictions on a batch/collection of examples; need to add to a list
img = (np.expand_dims(img, 0))
print(img.shape)

#predict iamge
predictions_single = model.predict(img)

print(predictions_single)

plot_value_array(0, predictions_single, test_labels)
_ = plt.xticks(range(10), class_names, rotation=45)

np.argmax(predictions_single[0])

plt.show()
