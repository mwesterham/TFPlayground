from abc import ABC, abstractmethod

class TFTrainer(ABC):
    """
    An abstraction for training tensorflow models
    """

    def __init__(self, definition):
        self.definition = definition

    def run(self):
        """
        Trains a tf model according to class definition

        :return: tuple like (processed_data, trained_tf_model, history)
        """
        data = self._get_data()
        processed_data = self._preprocess(data)
        tf_model = self._get_tf_model()
        history = self._train_tf_model(tf_model, processed_data)
        return processed_data, tf_model, history

    @abstractmethod
    def _get_data(self):
        """
        Retrieve the training data, can be in any format but must be returned
        by the function

        :return: unprocessed data
        """
        pass

    @abstractmethod
    def _preprocess(self, data):
        """
        Using the data retrieved beforehand, parse and preprocess it into the desired format.
        Must return the preprocessed data

        :param data: unprocessed data
        :return: preprocessed data
        """
        pass

    @abstractmethod
    def _get_tf_model(self):
        """
        Build the tensorflow model that we will train with

        :return: a tensorflow model to train with
        """
        pass

    @abstractmethod
    def _train_tf_model(self, model, processed_data):
        """
        Train the model with the given data

        :param model: the tf model to train with
        :param processed_data: the processed data to train on
        :return: a history object obtained from training
        """
        pass


class ModelOperator(ABC):
    """
    An abstraction for operating with tensorflow models
    """

    def __init__(self, model, definition=None):
        self.model = model
        self.definition = definition

    @abstractmethod
    def evaluate(self, params):
        """
        Performs evaluation of the model with given params and prints any necessary data

        :param params: contains any necessary data to perform analysis
        :return: defined metrics
        """
        pass

    @abstractmethod
    def use(self, input, config=None):
        """
        Uses the model with the given input

        :param input: input data into the model
        :param config: defines any configuration needed to operate with the model
        :return: the output of the model
        """
        pass

    @abstractmethod
    def plot(self, history):
        """
        Helper function to plot the model's performance

        :param history: history object obtained from the model training
        """
        pass
