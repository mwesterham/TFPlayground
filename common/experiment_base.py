from abc import ABC, abstractmethod
import time

class Experiment(ABC):
    """
    Base class used to define what an experiment entails
    """

    def __init__(self):
        self.runtime = 0

    def run(self):
        """
        A wrapper function to time an experiment
        """

        start_time = time.time()
        result = self._execute()
        end_time = time.time()
        self.runtime = end_time - start_time
        return result

    @abstractmethod
    def _execute(self):
        """
        A simple empty function to execute an experiment
        """
        pass