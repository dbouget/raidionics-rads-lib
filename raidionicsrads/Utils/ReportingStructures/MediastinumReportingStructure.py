import logging
import traceback
import os
import numpy as np
import operator
import json
import pandas as pd
import collections
from ..configuration_parser import ResourcesConfiguration


class MediastinumReportingStructure:
    """

    """
    _unique_id = None  # Internal unique identifier for the report
    _radiological_volume_uid = None  # Parent CT/MRI volume to which the report is attached
    _output_folder = None
    _lymph_nodes_count = None
    _statistics = {}

    def __init__(self, id: str, parent_uid: str, output_folder: str):
        """
        """
        self.__reset()
        self._unique_id = id
        self._radiological_volume_uid = parent_uid
        self._output_folder = output_folder
        self._statistics['LymphNodes'] = {}
        self._statistics['LymphNodes']['Overall'] = LymphNodeStatistics()

    def __reset(self):
        """
        All objects share class or static variables.
        An instance or non-static variables are different for different objects (every object has a copy).
        """
        self._unique_id = None
        self._radiological_volume_uid = None
        self._output_folder = None
        self._lymph_nodes_count = None
        self._statistics = {}

    def setup(self, tumor_elements: int) -> None:
        self._lymph_nodes_count = tumor_elements
        self._statistics['LymphNodes'] = {}
        self._statistics['LymphNodes']['Overall'] = None
        for p in range(tumor_elements):
            self._statistics['LymphNodes'][str(p+1)] = LymphNodeStatistics()

    def to_txt(self) -> None:
        """

        Exporting the computed tumor characteristics and standardized report.

        Parameters
        ----------

        Returns
        -------
        None
        """
        try:
            filename = os.path.join(self._output_folder, "mediastinum_clinical_report.txt")
            logging.info("Exporting mediastinum-parameters to text in {}.".format(filename))
            pfile = open(filename, 'a')
            pfile.write('########### Raidionics clinical report ###########\n')

            pfile.close()
        except Exception as e:
            logging.error("Mediastinum-parameters export to text failed with {}".format(traceback.format_exc()))
        return

    def to_json(self) -> None:
        try:
            filename = os.path.join(self._output_folder, "mediastinum_clinical_report.json")
            logging.info("Exporting mediastinum-parameters to json in {}.".format(filename))
            param_json = {}
            param_json['Overall'] = {}
            param_json['Overall']['Lymphnodes_count'] = self._lymph_nodes_count

            param_json['LymphNodes'] = {}
            for p in range(self._lymph_nodes_count):
                tumor_component = str(p + 1)
                param_json['LymphNodes'][tumor_component] = {}
                param_json['LymphNodes'][tumor_component]['Volume'] = self._statistics['LymphNodes'][
                    tumor_component].volume
                param_json['LymphNodes'][tumor_component]['Axis_diameters'] = self._statistics['LymphNodes'][
                    tumor_component].axis_diameters

            with open(filename, 'w', newline='\n') as outfile:
                json.dump(param_json, outfile, indent=4, sort_keys=True)
        except Exception as e:
            logging.error("Mediastinum-parameters export to json failed with {}".format(traceback.format_exc()))
        return

    def to_csv(self) -> None:
        # @TODO. To implement.
        try:
            filename = os.path.join(self._output_folder, "mediastinum_clinical_report.csv")
            logging.info("Exporting mediastinum-parameters to csv in {}.".format(filename))

            # values_df = pd.DataFrame(np.asarray(values).reshape((1, len(values))), columns=column_names)
            # values_df.to_csv(filename, index=False)
        except Exception as e:
            logging.error("Mediastinum-parameters export to csv failed with {}".format(traceback.format_exc()))


class LymphNodeStatistics:
    def __init__(self):
        self.laterality = None
        self.volume = None
        self.axis_diameters = []
        self.stations_overlap = {}
