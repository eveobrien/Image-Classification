import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import datasets, layers, models #Keras is the dataset

#Step 1 - Loading and Preparing data

#Loading data from the dataset and splitting training and testing data
(training_images, training_labels), (testing_images, testing_labels) = datasets.cifar10.load_data()

#Preparing data - scaling data down to 0-1 to make it more convinient to work with
training_images, testing_images = training_images / 255, testing_images / 255

#Define class / label names and visualise some images from the keras dataset
class_names = ['Plane', 'Car', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']

'''
#Using a 4x4 grid to visualise the 16 images to see what they look like
for i in range(16):
    plt.subplot(4,4,i+1) #i+1 adds the images and iterates
    plt.xticks([])
    plt.yticks([])
    plt.imshow(training_images[i], cmap=plt.cm.binary) #shows training images.
    plt.xlabel(class_names[training_labels[i][0]]) #xlabel will be the training labels class name. E.g. plane has index of 1.

plt.show() #displays the images
'''

#Step 2 - Building and Training the model.

#reducing the amount of images we are training into the network to the first 20k training and 4k testing to save time.
training_images = training_images[:20000]
training_labels = training_labels[:20000]
testing_images = testing_images[:4000]
testing_labels = testing_labels[:4000]

#building the network using a Convultion Matrix with 32 neurons with relu as our activation function
model = models.Sequential()

#Convolutional and MaxPooling layers. Convolutional layers filters for features in an image e.g. horse has long legs. MaxPooling reduces the image to its essential information
model.add(layers.Conv2D(32, (3,3), activation='relu', input_shape=(32, 32, 3)))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(32, (3,3), activation='relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64, (3,3), activation='relu'))

#Flatten the input layer to make it 1D
model.add(layers.Flatten())

#Add 2 Dense / Hidden Layers
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(64, activation='softmax')) #Scales them so they add up to 1 so you get a  probability.

#Compile them
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

#Fit the model on training images and labels with 10 epochs
model.fit(training_images, training_labels, epochs=10, validation_data=(testing_images, testing_labels))

#Step 3 - Test, Evaluate and Save the model

loss, accuracy = model.evaluate(testing_images, testing_labels)
print(f"Loss: {loss}") #Loss - how wrong thr model is
print(f"Accuracy: {accuracy}") #Accuracy - % of testing examples that were identified correctly

model.save('image_classifier.model')

#Load the model so it doesn't have to be trained each time
model = models.load_model('image_classifier.model')

#Taking random license free images and classifying them with the model with cv and numpy

#Load the image - Give it a different image each time
img = cv.imread('horse.jpg')

#convert colour scheme to swap blue and red values
img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

plt.imshow(img, cmap=plt.cm.binary) #visualise image

prediction = model.predict(np.array([img]) / 255)
index = np.argmax(prediction) #argmax gives index of maximum neuron value
print(f'Prediction is {class_names[index]}')





