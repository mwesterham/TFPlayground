import tensorflow as tf
from common.experiment_base import Experiment
from image_generator_experiment.model.dcgan_trainer import DCGANTrainer, GeneratorOperator, DiscriminatorOperator


class DCGANExperiment(Experiment):
    def __init__(self, config={}):
        # the inputted config takes precedence over the default
        config = {**{

        }, **config}

        self.config = config

        super().__init__()

    def _execute(self):
        mnist_dataset = tf.keras.datasets.mnist.load_data()
        trainer = DCGANTrainer(mnist_dataset, config={
            'EPOCHS': 200,
            'checkpoint_frequency': 1,
            'restore_ckpt': True,
        })

        trainer.run()


def run_dcgan_experiment():
    experiment = DCGANExperiment()
    experiment.run()
