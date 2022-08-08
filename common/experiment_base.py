from abc import ABC, abstractmethod


class Experiment(ABC):
    """
    Base class used to define what an experiment entails
    """

    @abstractmethod
    def execute(self):
        """
        A simple empty function to execute an experiment
        """
        pass
