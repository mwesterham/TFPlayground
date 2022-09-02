# Machine Learning Sandbox

This repository serves as demonstration of TensorFlow model implementations and ML concepts in the form of different
experiments. A series of base classes are additionally provided in order to facilitate each experiment and generify the
training process. An explanation of each experiment is provided below. TensorFlow 2.8.2 is used in each experiment.

- [**Image Classifier Experiments**](image_classifier_experiment/Image%20Classifier%20README.md)

The image classifier experiments are based off of Tensorflow's documentation
found [here](https://www.tensorflow.org/tutorials/keras/classification). In this experiment we implement a model that
classifies images into different classes for two datasets: Fashion MNIST and MNIST. Once these models were trained, they
were evaluated with the test dataset provided by Tensorflow and a small custom dataset I compiled myself for each. The
results from these two datasets are compared and analyzed.

- [**Text Binary Classifier Experiment**](text_binary_classifier_experiment/Text%20Binary%20Classifier%20README.md)

The text classifier experiment is based off of Tensorflow's documentation
found [here](https://www.tensorflow.org/tutorials/keras/text_classification). The aim of this experiment is
to train a binary classifier that performs sentiment analysis on the Large Movie Review Dataset (aclImdb).
Additionally, I provide model checkpointing and workflow abstraction to demonstrate these common ML processes.

