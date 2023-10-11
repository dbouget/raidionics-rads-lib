from aenum import Enum, unique
import logging
import traceback
import os
import numpy as np
import operator
import json
import pandas as pd
import collections
from ..configuration_parser import ResourcesConfiguration
from ..io import generate_cortical_structures_labels_for_slicer, generate_subcortical_structures_labels_for_slicer


@unique
class ResectionCategoryType(Enum):
    """

    """
    _init_ = 'value string'

    SupR = 0, 'Supramaximal resection of CE tumor'
    ComR = 1, 'Complete resection of CE tumor'
    NeaR = 2, 'Near total resection of CE tumor'
    SubR = 3, 'Subtotal resection of CE tumor'
    ParR = 4, 'Partial resection of CE tumor'

    def __str__(self):
        return self.string


class NeuroSurgicalReportingStructure:
    """
    Reporting at a single timestamp with characteristics/features for the tumor.
    """
    _unique_id = None  # Internal unique identifier for the report
    _output_folder = None
    _statistics = None

    def __init__(self, id: str, output_folder: str):
        """
        """
        self.__reset()
        self._unique_id = id
        self._output_folder = output_folder
        self._statistics = SurgicalStatistics()

    def __reset(self):
        """
        All objects share class or static variables.
        An instance or non-static variables are different for different objects (every object has a copy).
        """
        self._unique_id = None
        self._output_folder = None
        self._statistics = None

    def setup(self) -> None:
        pass

    def to_txt(self) -> None:
        """

        Exporting the computed tumor characteristics and standardized report in filename.

        Parameters
        ----------

        Returns
        -------
        None
        """
        try:
            filename = os.path.join(self._output_folder, "neuro_surgical_report.txt")
            logging.info("Exporting neuro-parameters to text in {}.".format(filename))
            pfile = open(filename, 'a')
            pfile.close()
        except Exception as e:
            logging.error("Neuro-parameters export to text failed with {}".format(traceback.format_exc()))
        return

    def to_json(self) -> None:
        try:
            param_json = {}
            filename = os.path.join(self._output_folder, "neuro_surgical_report.json")
            logging.info("Exporting neuro surgical parameters to json in {}.".format(filename))
            param_json["preop_volume"] = self._statistics.preop_volume
            param_json["postop_volume"] = self._statistics.postop_volume
            param_json["eor"] = self._statistics.extent_of_resection
            param_json["resection_category"] = str(self._statistics.resection_category)
            with open(filename, 'w', newline='\n') as outfile:
                json.dump(param_json, outfile, indent=4, sort_keys=True)
        except Exception as e:
            logging.error("Neuro-parameters export to json failed with {}".format(traceback.format_exc()))

        return

    def to_csv(self) -> None:
        try:
            filename = os.path.join(self._output_folder, "neuro_surgical_report.csv")
            logging.info("Exporting neuro-parameters to csv in {}.".format(filename))
        except Exception as e:
            logging.error("Neuro-parameters export to csv failed with {}".format(traceback.format_exc()))


class SurgicalStatistics:
    """
    Specific class for holding the surgical features/characteristics
    """
    def __init__(self):
        self.preop_volume = None
        self.postop_volume = None
        self.extent_of_resection = None
        self.classification = None
        self.resection_category = None
