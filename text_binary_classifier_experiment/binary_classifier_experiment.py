import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import losses
import matplotlib.pyplot as plt
import os
import re
import shutil
import string
from pathlib import Path

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
        if(self.config["download_data"]):
            url = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
            tf.keras.utils.get_file("aclImdb_v1", url,
                                              untar=True, cache_dir=self.config["cache_dir"],
                                              cache_subdir='')

        dataset_dir = Path(self.config["cache_dir"]) / 'aclImdb'
        train_dir = os.path.join(dataset_dir, 'train')
        print(os.listdir(train_dir))

        sample_file = os.path.join(train_dir, 'pos/1181_9.txt')
        with open(sample_file) as f:
            print(f.read())

def run_text_classifier_experiment():
    classifier1 = BinaryClassifierExperiment(config={
        "download_data": False
    })
    print("--- Runtime: %s seconds ---" % classifier1.runtime)
    classifier1.run()
    print("--- Runtime: %s seconds ---" % classifier1.runtime)

    classifier2 = BinaryClassifierExperiment(config={
        "download_data": True
    })
    classifier2.run()
    print(classifier2.runtime)
    print("--- Runtime: %s seconds ---" % classifier2.runtime)

