import tensorflow as tf
from image_classifier_experiment.classifier_experiment import ClassifierExperiment
import matplotlib.pyplot as plt


def run_classifier_experiment():
    # Based off of https://www.tensorflow.org/tutorials/keras/classification

    epoch_list = [0, 5, 10, 15, 20, 25, 30, 40, 50, 60, 70, 80, 100]
    experiment1_results = {
        'test': {
            'epochs': epoch_list,
            'loss': [],
            'accuracy': [],
        },
        'custom': {
            'epochs': epoch_list,
            'loss': [],
            'accuracy': [],
        }
    }
    experiment2_results = {
        'test': {
            'epochs': epoch_list,
            'loss': [],
            'accuracy': [],
        },
        'custom': {
            'epochs': epoch_list,
            'loss': [],
            'accuracy': [],
        }
    }

    for EPOCHS in epoch_list:
        # execute experiment with Fashion MNIST
        experiment1 = ClassifierExperiment({
            'EPOCHS': EPOCHS,
            'asset_opts': {
                'save_assets': True,
                'dir': './image_classifier_experiment/generated_assets/fashion_mnist/'
            },
            'dataset': tf.keras.datasets.fashion_mnist,
            'custom_image_dir': './image_classifier_experiment/custom images/fashion_mnist/*.jpg',
            'labels': [0, 8, 8, 3, 2, 1, 4, 5, 9, 7, 6],
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
        test_image_results1, custom_image_results1 = experiment1.execute()
        experiment1_results['test']['loss'].append(test_image_results1['loss'])
        experiment1_results['test']['accuracy'].append(test_image_results1['accuracy'])
        experiment1_results['custom']['loss'].append(custom_image_results1['loss'])
        experiment1_results['custom']['accuracy'].append(custom_image_results1['accuracy'])

        # execute experiment with MNIST
        experiment2 = ClassifierExperiment({
            'EPOCHS': EPOCHS,
            'asset_opts': {
                'save_assets': True,
                'dir': './image_classifier_experiment/generated_assets/mnist/'
            },
            'dataset': tf.keras.datasets.mnist,
            'custom_image_dir': './image_classifier_experiment/custom images/mnist/*.png',
            'labels': [0, 0, 1, 9, 8, 2, 4, 3, 5, 7, 6],
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
        test_image_results2, custom_image_results2 = experiment2.execute()
        experiment2_results['test']['loss'].append(test_image_results2['loss'])
        experiment2_results['test']['accuracy'].append(test_image_results2['accuracy'])
        experiment2_results['custom']['loss'].append(custom_image_results2['loss'])
        experiment2_results['custom']['accuracy'].append(custom_image_results2['accuracy'])

    # plot results for test data
    plt.plot('epochs', 'loss', data=experiment1_results['test'], label='Fashion MNIST')
    plt.plot('epochs', 'loss', data=experiment2_results['test'], label='MNIST')
    plt.xlabel('EPOCHS')
    plt.ylabel('Loss')
    plt.title('Test Data Fashion MNIST vs MNIST')
    plt.legend()
    plt.savefig("./image_classifier_experiment/generated_assets/Test Data Fashion MNIST vs MNIST.png")
    plt.show()

    # plot results for custom data
    plt.plot('epochs', 'loss', data=experiment1_results['custom'], label='Fashion MNIST')
    plt.plot('epochs', 'loss', data=experiment2_results['custom'], label='MNIST')
    plt.xlabel('EPOCHS')
    plt.ylabel('Loss')
    plt.title('Custom Data Fashion MNIST vs MNIST')
    plt.legend()
    plt.savefig("./image_classifier_experiment/generated_assets/Custom Data Fashion MNIST vs MNIST.png")
    plt.show()


if __name__ == '__main__':
    run_classifier_experiment()
