import json
from abc import ABC, abstractmethod


class AbstractPipelineStep(ABC):
    _step_json = {}
    _step_description = None

    def __init__(self, step_json: dict) -> None:
        self.__reset()
        self._step_json = step_json
        self._step_description = step_json["description"]

    def __reset(self):
        """
        All objects share class or static variables.
        An instance or non-static variables are different for different objects (every object has a copy).
        """
        self._step_json = {}
        self._step_description = None

    @abstractmethod
    def setup(self, patient_parameters):
        pass

    @abstractmethod
    def execute(self):
        pass

    # @TODO. Should there be a step assess method, before executing the pipeline, ensuring all inputs are available
    # for all steps.
