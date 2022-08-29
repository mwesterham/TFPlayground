import tensorflow as tf
import shutil
from pathlib import Path

from text_binary_classifier_experiment.model.text_classifier import TextClassifierTrainer, TextClassifierOperator
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

        # Instantiate and run the trainer
        TRAINER = TextClassifierTrainer(dataset_dir)
        processed_data, trained_tf_model, history = TRAINER.run()

        # Evaluate the model
        OPERATOR = TextClassifierOperator(trained_tf_model)
        loss, accuracy = OPERATOR.evaluate(processed_data)

        print("Loss: ", loss)
        print("Accuracy: ", accuracy)

        predictions = OPERATOR.use(config={
            'vectorize_layer': TRAINER.definition['vectorize_layer']
        }, input_data=[
            'Inception is such a mind-blowing, storytelling masterpiece and Christopher Nolan\'s 1# film while also being one of the greatest science fiction action movies ever made. While confusing during the first scene with Dom Cobb and the elderly Saito, It has a fascinating plot and theme of planting an idea; convincing the subject\'s mind that they\'re not dreaming is the ultimate result of Mal Cobb\'s inception. I highly praise Hans Zimmer\'s music, the explanations, the visuals, the sound, Nolan\'s direction, and the performance of Leonardo DiCaprio, Tom Hardy, Ken Watanabe, and Marion Cotillard.',
            'Nolan once said, "chase your reality". That is all well and fine. The movie fails to distinguish the reality and dream. And I am personally fine with a movie exploring such extents. What is disappointing is the way things are perceived by people who after watching this movie seem to believe that we choose our reality as in even dreams could be reality. ',

        ])
        print(predictions)

        OPERATOR.plot(history)

    def __print_file(self, filepath):
        """Helper function that can print the contents of an individual file"""

        with open(filepath) as f:
            print(f.read())


def run_text_classifier_experiment():
    classifier1 = BinaryClassifierExperiment(config={
        "download_data": False
    })
    classifier1.run()
    print("--- Runtime: %s seconds ---" % classifier1.runtime)
