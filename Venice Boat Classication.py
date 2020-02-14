"""
Author: Ahmed Elhelow                           1823229
Help was given by Hossam Arafat and Michele Cipriano
"""

#==============================================================================

# Difine libraries
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import sklearn
from sklearn import preprocessing, metrics
from sklearn.metrics import confusion_matrix
import random
from random import shuffle
from heapq import nlargest

# Importing and mount google drive
from google.colab import drive
drive.mount('/content/gdrive',force_remount=False)

# Importing TF Learn
import tensorflow as tf
import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression

# Importing keras
import keras

#==============================================================================

# Set seed for random numbers
random.seed(1)

# Define directories and required files
DIR = "/content/gdrive/My Drive/"
TRAIN_DIR = "/content/gdrive/My Drive/sc5/"
DB_info = "/content/gdrive/My Drive/sc5/DBinfo.txt"
TEST_DIR = "/content/gdrive/My Drive/sc5-test/"
GROUND_TRUTH = "/content/gdrive/My Drive/sc5-test/ground_truth.txt"

# Define the image size and learning rate for the CNN
IMG_WIDTH_COLOR = 224
IMG_HEIGHT_COLOR = 224
IMG_WIDTH_GRAY = 120
IMG_HEIGHT_GRAY = 120
LEARN_RATE = 0.001

#==============================================================================

# Explore the Boats Training Data Set to check for the available classes
training_classes = [f.name for f in os.scandir(TRAIN_DIR) if f.is_dir()]

# Explore the Boats Testing Data Set to check for the available classes
test_data_dict = dict()
freq_of_each_class = dict()
classes = list()
testing_classes = list()

# Opening the Ground Truth File and Reading its contents
with open(GROUND_TRUTH, 'r') as f:
    f_content = f.read().splitlines()
for line in f_content:
    filename, label = line.split(";")
    label = label.replace(" ", "")  # Format the class label by removing spaces
    label = label.replace(":", "")  # Format the class label by removing colons
    classes.append(label)
    if label not in testing_classes:
        testing_classes.append(label)

# Count the frequency of occurrence of each class/label in the Testing Data Set
for cls in classes:
    freq_of_each_class[cls] = freq_of_each_class.get(cls, 0) + 1

print("freq_of_each_class: \n", freq_of_each_class)

#==============================================================================

# Find the Common Classes
common_classes = [element for element in training_classes if element in \
                  testing_classes]

freq_of_common_class = dict()
for cls in classes:
    if cls in common_classes:
        freq_of_common_class[cls] = freq_of_common_class.get(cls, 0) + 1
        
# Pick 3 Classes with the most number of images from the common Classes
NUM_CLASSES = 3
counts = nlargest(NUM_CLASSES, freq_of_common_class.values())
classes_to_be_considered = [key for key, value in \
                            freq_of_common_class.items() if value in counts]

# Transform labels from a list of strings to a list of Numbers
numerical_form_classes = np.asarray([classes_to_be_considered.index(t) \
                                     for t in classes_to_be_considered])
classes_onehot = preprocessing.OneHotEncoder(sparse=False). \
                 fit_transform(numerical_form_classes.reshape(-1, 1))

# Dictionary with Class name and Corresponding label
corres_label = dict(zip(classes_to_be_considered, classes_onehot))
print("Dictionary with Class name and Corresponding label: \n", corres_label)

#==============================================================================

X_color = []
X_gray = []
Y = []

for folder in training_classes:
    if folder in classes_to_be_considered:
        path = TRAIN_DIR + folder
        files = os.listdir(path)
        for img in files:
#             print(":", end='')
            label = corres_label[folder]
            img_path = os.path.join(path, img)
            img = cv2.resize(cv2.imread(img_path, cv2.IMREAD_UNCHANGED), \
                             (IMG_HEIGHT_COLOR, IMG_WIDTH_COLOR))
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img_gray = cv2.resize(img_gray,(IMG_HEIGHT_GRAY, IMG_WIDTH_GRAY))
            X_color.append(img)
            X_gray.append(img_gray)
            Y.append(label)
#         print()

training_data = list(zip(X_color, X_gray, Y))
shuffle(training_data)
X_color, X_gray, Y = zip(*training_data)

#==============================================================================

# Build the Testing Data Set
considered_test_filenames = list()
considered_test_labels = list()

with open(GROUND_TRUTH, 'r') as f:
    f_content = f.read().splitlines()
for line in f_content:
    filename, label = line.split(";")
    label = label.replace(" ", "")  # Format the class label by removing spaces
    label = label.replace(":", "")  # Format the class label by removing colons
    if label in classes_to_be_considered:
        considered_test_filenames.append(filename)
        considered_test_labels.append(corres_label[label])

testing_corres_label = dict(zip(considered_test_filenames, \
                                considered_test_labels))

# Building the model's testing data set
x_test_color = []
x_test_gray = []
y_test = []
for test_img in os.listdir(TEST_DIR):
    if test_img == "Thumbs.db":
        continue
    if test_img in testing_corres_label.keys():
#         print(":", end='')
        test_data_label = testing_corres_label[test_img]
        test_img_path = os.path.join(TEST_DIR, test_img)
        test_img = cv2.resize(cv2.imread(test_img_path, cv2.IMREAD_UNCHANGED) \
                              ,(IMG_HEIGHT_COLOR, IMG_WIDTH_COLOR))
        test_img_gray = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)
        test_img_gray = cv2.resize(test_img_gray,(IMG_HEIGHT_GRAY, \
                                                  IMG_WIDTH_GRAY))
        x_test_color.append(test_img)
        x_test_gray.append(test_img_gray)
        y_test.append(test_data_label)
# print()

testing_data = list(zip(x_test_color, x_test_gray, y_test))
shuffle(testing_data)
x_test_color, x_test_gray, y_test = zip(*testing_data)

#==============================================================================

# Prepare the Testing and Training Data that will be used to train the Network
X_color = np.array(X_color)
X_gray = np.array(X_gray).reshape(-1, IMG_WIDTH_GRAY, IMG_HEIGHT_GRAY, 1)
Y = np.array(Y)

x_test_color = np.array(x_test_color)
x_test_gray = np.array(x_test_gray).reshape(-1,IMG_WIDTH_GRAY, \
                                            IMG_HEIGHT_GRAY, 1)
y_test = np.array(y_test)

#==============================================================================

# Load pretrained VGG-16 model:
vgg16_tmp = keras.applications.vgg16.VGG16()

# Build a new model similar to the previous one with the right number
# of nodes on the softmax layer and make all the other layer not trainable:
vgg16_model = keras.models.Sequential()

for layer in vgg16_tmp.layers:
    vgg16_model.add(layer)

vgg16_model.layers.pop()

for layer in vgg16_model.layers:
    layer.trainable = False

vgg16_model.add(keras.layers.Dense(NUM_CLASSES, activation="softmax"))

#==============================================================================

# Train the network with SGD with momentum:
vgg16_model.compile(keras.optimizers.SGD(lr=1e-3, decay=1e-6, 
                                         momentum=0.9, nesterov=True),
                    loss="categorical_crossentropy",
                    metrics=["accuracy"])

vgg16_model.fit(X_color, Y, batch_size=32, epochs=5)

#==============================================================================

y_pred = vgg16_model.predict(x_test_color)
y_pred = np.argmax(y_pred, axis=1)
y_test_orig = np.argmax(y_test, axis=1)

print("Accuracy: " + str(sklearn.metrics.accuracy_score(y_test_orig, y_pred)))
print("Confusion matrix: \n", confusion_matrix(y_test_orig, y_pred))

#==============================================================================

tf.reset_default_graph()

# Model 2
# Build the model and train it
convnet = input_data(shape=[None, IMG_WIDTH_GRAY, IMG_HEIGHT_GRAY, 1], \
                     name='input')

convnet = conv_2d(convnet, 32, 2, activation='relu')
convnet = max_pool_2d(convnet, 2)

convnet = conv_2d(convnet, 64, 2, activation='relu')
convnet = max_pool_2d(convnet, 2)

convnet = conv_2d(convnet, 32, 2, activation='relu')
convnet = max_pool_2d(convnet, 2)

convnet = conv_2d(convnet, 64, 2, activation='relu')
convnet = max_pool_2d(convnet, 2)

convnet = conv_2d(convnet, 32, 2, activation='relu')
convnet = max_pool_2d(convnet, 2)

convnet = conv_2d(convnet, 64, 2, activation='relu')
convnet = max_pool_2d(convnet, 2)

# You have 2 fully connected layers here
# One is the fully connected and the other is the o/p layer
convnet = fully_connected(convnet, 1024, activation='relu')
convnet = dropout(convnet, 0.8)

convnet = fully_connected(convnet, 3, activation='softmax') # Outplut layer
convnet = regression(convnet, optimizer='adam', learning_rate=LEARN_RATE, 
                     loss='categorical_crossentropy', name='targets')

# Experiment changing the learning rate and
model = tflearn.DNN(convnet, tensorboard_dir='log_1')

#==============================================================================

model.fit({'input': X_gray}, {'targets': Y}, n_epoch=5, \
          validation_set=({'input': x_test_gray}, {'targets': y_test}), \
          snapshot_step=50, show_metric=True, \
          run_id="boatClassifier-6conv-0.001LR")

print("Model 2 Trained!")

#==============================================================================

predicted = list()
truth = list()

for _, the_img, the_img_label in testing_data:
    the_image = the_img.reshape(IMG_WIDTH_GRAY, IMG_HEIGHT_GRAY, 1)
    model_out = model.predict([the_image])[0]
    
    predicted.append(np.argmax(model_out))
    truth.append(np.argmax(the_img_label))

print("The Second Model: ")
print("Accuracy: " + str(sklearn.metrics.accuracy_score(truth, predicted)))
print("Confusion matrix: \n", confusion_matrix(truth, predicted))

#==============================================================================
