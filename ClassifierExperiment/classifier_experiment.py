from common.experiment_base import Experiment
from ClassifierExperiment.model.classifier import ClassifierTrainer, ClassifierOperator
from PIL import Image, ImageOps
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
import math


class ClassifierExperiment(Experiment):
    def __init__(self, config={}):
        config = {**{
            'class_names': [
                'T-shirt/top', # 0
                'Trouser', # 1
                'Pullover', # 2
                'Dress', # 3
                'Coat', # 4
                'Sandal', # 5
                'Shirt', # 6
                'Sneaker', # 7
                'Bag', # 8
                'Ankle boot' # 9
            ],
            'custom_image_dir': 'C:\\Users\\mwest\\Documents\\Programming\Python Projects\\TensorFlowPlayground\\ClassifierExperiment\\custom images\\*.jpg',
            'labels': [0, 8, 8, 3, 2]
        }, **config}

        self.config = config

    def execute(self):
        TRAINER = ClassifierTrainer(self.load_data(), config={
            'input_shape': (28, 28),
            'nodes': 128,
            'EPOCHS': 100,
            'num_classes': len(self.config['class_names'])
        })
        (data, processed_data, trained_tf_model) = TRAINER.run()

        _, (test_images, test_labels) = data
        OPERATOR = ClassifierOperator(trained_tf_model)
        OPERATOR.evaluate((test_images, test_labels))

        # use and plot test images from dataset
        predictions_array = OPERATOR.use(test_images)
        self.plot_joint_predictions(predictions_array[0], test_labels[0], test_images[0])
        self.plot_multi_predicionts(predictions_array, test_labels, test_images, 15)

        # use and plot custom images from internet
        # forward declaration of all images
        custom_imgs = []
        for f in sorted(glob.glob(self.config['custom_image_dir']), key=os.path.basename):
            img = Image.open(f)
            newsize = (28, 28)
            img = img.resize(newsize)
            img = ImageOps.grayscale(img)
            img = ImageOps.invert(img)

            custom_imgs.append(np.asarray(img))
        custom_imgs = np.array(custom_imgs)

        # correct label indexed by photo name
        correct_labels = self.config['labels']
        custom_predictions_array = OPERATOR.use(custom_imgs)
        self.plot_multi_predicionts(custom_predictions_array, correct_labels, custom_imgs, len(correct_labels))

    def load_data(self):
        fashion_mnist = tf.keras.datasets.fashion_mnist
        return fashion_mnist.load_data()

    def display_image_grid(self, images_arr, labels_arr, class_names):
        plt.figure(figsize=(10, 10))
        for i in range(25):
            plt.subplot(5, 5, i + 1)
            plt.xticks([])
            plt.yticks([])
            plt.grid(False)
            plt.imshow(images_arr[i], cmap=plt.cm.binary)
            plt.xlabel(class_names[labels_arr[i]])
        plt.show()

    def plot_image(self, predictions_array, true_label, img):
        plt.grid(False)
        plt.xticks([])
        plt.yticks([])

        plt.imshow(img, cmap=plt.cm.binary)

        predicted_label = np.argmax(predictions_array)
        if predicted_label == true_label:
            color = 'blue'
        else:
            color = 'red'

        plt.xlabel("{} {:2.0f}% ({})".format(self.config['class_names'][predicted_label],
                                             100 * np.max(predictions_array),
                                             self.config['class_names'][true_label]),
                   color=color)

    def plot_value_array(self, predictions_array, true_label):
        plt.grid(False)
        plt.xticks(range(10))
        plt.yticks([])
        thisplot = plt.bar(range(10), predictions_array, color="#777777")
        plt.ylim([0, 1])
        predicted_label = np.argmax(predictions_array)

        thisplot[predicted_label].set_color('red')
        thisplot[true_label].set_color('blue')

    def plot_joint_predictions(self, prediction, test_label, test_image):
        plt.figure(figsize=(6, 3))
        plt.subplot(1, 2, 1)
        self.plot_image(prediction, test_label, test_image)
        plt.subplot(1, 2, 2)
        self.plot_value_array(prediction, test_label)
        plt.show()

    def plot_multi_predicionts(self, predictions, test_labels, test_images, num_images):
        # Plot the first X test images, their predicted labels, and the true labels.
        # Color correct predictions in blue and incorrect predictions in red.
        num_rows = math.ceil(num_images/3)
        num_cols = 3
        plt.figure(figsize=(2 * 2 * num_cols, 2 * num_rows))
        for i in range(num_images):
            plt.subplot(num_rows, 2 * num_cols, 2 * i + 1)
            self.plot_image(predictions[i], test_labels[i], test_images[i])
            plt.subplot(num_rows, 2 * num_cols, 2 * i + 2)
            self.plot_value_array(predictions[i], test_labels[i])
        plt.tight_layout()
        plt.show()
