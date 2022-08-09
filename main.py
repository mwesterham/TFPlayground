import tensorflow as tf
from image_classifier_experiment.classifier_experiment import ClassifierExperiment

if __name__ == '__main__':
    # Based off of https://www.tensorflow.org/tutorials/keras/classification
    experiment1 = ClassifierExperiment({
        'EPOCHS': 10,
        'asset_opts': {
            'save_assets': True,
            'dir': './image_classifier_experiment/generated_assets/fashion_mnist/'
        },
        'dataset': tf.keras.datasets.fashion_mnist,
        'custom_image_dir': './image_classifier_experiment/custom images/fashion_mnist/*.jpg',
        'labels': [0, 8, 8, 3, 2],
        'class_names': [
            'T-shirt/top',  # 0
            'Trouser',  # 1
            'Pullover',  # 2
            'Dress',  # 3
            'Coat',  # 4
            'Sandal',  # 5
            'Shirt',  # 6
            'Sneaker',  # 7
            'Bag',  # 8
            'Ankle boot'  # 9
        ],
    })
    experiment1.execute()

    experiment2 = ClassifierExperiment({
        'EPOCHS': 10,
        'asset_opts': {
            'save_assets': True,
            'dir': './image_classifier_experiment/generated_assets/mnist/'
        },
        'dataset': tf.keras.datasets.mnist,
        'custom_image_dir': './image_classifier_experiment/custom images/mnist/*.png',
        'labels': [0, 0, 1, 9, 8],
        'class_names': [
            '0',  # 0
            '1',  # 1
            '2',  # 2
            '3',  # 3
            '4',  # 4
            '5',  # 5
            '6',  # 6
            '7',  # 7
            '8',  # 8
            '9'  # 9
        ],
    })
    experiment2.execute()
