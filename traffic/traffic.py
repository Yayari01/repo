import cv2
import numpy as np
import os
import sys
import tensorflow as tf

from sklearn.model_selection import train_test_split

EPOCHS = 15
IMG_WIDTH = 30
IMG_HEIGHT = 30
NUM_CATEGORIES = 43
TEST_SIZE = 0.4


def main():

    # Check command-line arguments
    if len(sys.argv) not in [2, 3]:
        sys.exit("Usage: python traffic.py data_directory [model.h5]")

    # Get image arrays and labels for all image files
    images, labels = load_data(sys.argv[1])

    # Split data into training and testing sets
    labels = tf.keras.utils.to_categorical(labels)
    x_train, x_test, y_train, y_test = train_test_split(
        np.array(images), np.array(labels), test_size=TEST_SIZE
    )

    # Get a compiled neural network
    model = get_model()

    # Fit model on training data
    model.fit(x_train, y_train, epochs=EPOCHS)

    # Evaluate neural network performance
    model.evaluate(x_test,  y_test, verbose=2)

    # Save model to file
    if len(sys.argv) == 3:
        filename = sys.argv[2]
        model.save(filename)
        print(f"Model saved to {filename}.")


def load_data(data_dir):
    """
    Load image data from directory `data_dir`.

    Assume `data_dir` has one directory named after each category, numbered
    0 through NUM_CATEGORIES - 1. Inside each category directory will be some
    number of image files.

    Return tuple `(images, labels)`. `images` should be a list of all
    of the images in the data directory, where each image is formatted as a
    numpy ndarray with dimensions IMG_WIDTH x IMG_HEIGHT x 3. `labels` should
    be a list of integer labels, representing the categories for each of the
    corresponding `images`.
    """

    # getting the name of folders by calling os.listdir() and initialising the empty lists for returning the processed images along with the correct labels
    folder_names = os.listdir(data_dir)

    image_list = []

    label_list = []

    # getting each folder name from the list of folder_names
    for folder in folder_names:

        # Creating the folder path with os.path.join which gets passed onto the os listdir to get the names of images
        folder_path = os.path.join(data_dir, folder)

        if not os.path.isdir(folder_path):

            continue

        folder_img = os.listdir(folder_path)

        # iterating over each image name
        for img in folder_img:

            # constructing the path to get the actual data based of the name
            img_path = os.path.join(data_dir, folder, img)

            # passing the image's path into the image reader which gives us the actual image data
            image = cv2.imread(img_path)

            # then we use cv2's image resizing method to resize our images based of the measurements held in IMG_WIDTH , IMG_HEIGHT
            resized_img = cv2.resize(image, (IMG_WIDTH, IMG_HEIGHT))

            # appending the resized image to the resized image list that will get returned at the end of the function
            image_list.append(resized_img)

            # converting the name of the folder which is a string to an integer that will be returned as the label for the data
            label_list.append(int(folder))

    return image_list, label_list


def get_model():
    """
    Returns a compiled convolutional neural network model. Assume that the
    `input_shape` of the first layer is `(IMG_WIDTH, IMG_HEIGHT, 3)`.
    The output layer should have `NUM_CATEGORIES` units, one for each category.
    """

    model = tf.keras.models.Sequential([

        # 1st layer taking a 30x30 RGB image as an input
        tf.keras.layers.Input(shape=(IMG_WIDTH, IMG_HEIGHT, 3)),

        # first conv2D layer detecting basic features, angles, colours
        tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation="relu"),

        # second conv2D layer detecting more detailed features, shapes , patterns
        tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation="relu"),

        # Pooling layer reducing dimensions
        tf.keras.layers.MaxPooling2D((2, 2)),

        # Flatten layer converting 2D features to 1D
        tf.keras.layers.Flatten(),

        # the dense layer learning features using 128 neurons
        tf.keras.layers.Dense(units=128, activation='relu'),

        # Using dropout to randomly turn off 40% of neurons during training to prevent overfitting
        tf.keras.layers.Dropout(0.4),

        # output layer producing probabilities based on which to categorize each image into one of the 43 traffic sign categories
        tf.keras.layers.Dense(NUM_CATEGORIES, activation="softmax")


    ])

    model.compile(optimizer="adam", loss='categorical_crossentropy', metrics=['accuracy'])

    model.summary()

    return model


if __name__ == "__main__":
    main()
