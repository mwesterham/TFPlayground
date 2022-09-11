import tensorflow as tf
import imageio.v2 as imageio
from os import listdir
from pathlib import Path
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
            'EPOCHS': 15,
            'checkpoint_frequency': 3,
            'restore_ckpt': True,
        })

        trainer.run()


def build_gif(img_dir):
    with imageio.get_writer(Path(img_dir).parent / 'movie.gif', mode='I') as writer:
        for filename in listdir(img_dir):
            filename = Path(img_dir) / filename
            print("appending {}".format(filename))
            image = imageio.imread(filename)
            writer.append_data(image)


def run_dcgan_experiment():
    experiment = DCGANExperiment()
    experiment.run()

    build_gif('./image_generator_experiment/cache/imgs/')