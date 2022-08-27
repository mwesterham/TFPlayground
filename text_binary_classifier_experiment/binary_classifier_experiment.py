import tensorflow as tf
import shutil
from pathlib import Path

from text_binary_classifier_experiment.model.text_classifier import TextClassifierTrainer
from common.experiment_base import Experiment


class BinaryClassifierExperiment(Experiment):
    def __init__(self, config={}):
        # the inputted config takes precedence over the default
        config = {**{
            "download_data": True,
            "dataset": 'aclImdb',
            "cache_dir": './text_binary_classifier_experiment/cache'
        }, **config}

        self.config = config

        super().__init__()

    def _execute(self):
        # download data or grab from cache dir
        if (self.config["download_data"]):
            url = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
            tf.keras.utils.get_file("aclImdb_v1", url,
                                    untar=True, cache_dir=self.config["cache_dir"],
                                    cache_subdir='')

            # remove this directory, it is unneeded
            remove_dir = Path(self.config["cache_dir"]) / 'aclImdb' / 'train' / 'unsup'
            shutil.rmtree(remove_dir)

        dataset_dir = Path(self.config["cache_dir"]) / 'aclImdb'
        trainer = TextClassifierTrainer(dataset_dir)
        (processed_data, trained_tf_model) = trainer.run()
        train_ds, val_ds, test_ds = processed_data

        loss, accuracy = trained_tf_model.evaluate(test_ds)

        print("Loss: ", loss)
        print("Accuracy: ", accuracy)

    def __print_file(self, filepath):
        with open(filepath) as f:
            print(f.read())


def run_text_classifier_experiment():
    classifier1 = BinaryClassifierExperiment(config={
        "download_data": False
    })
    classifier1.run()
    print("--- Runtime: %s seconds ---" % classifier1.runtime)
