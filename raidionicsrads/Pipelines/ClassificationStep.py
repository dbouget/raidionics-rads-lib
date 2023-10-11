import os
import shutil

import nibabel as nib
import pandas as pd
import logging
import configparser
import traceback
from ..Utils.configuration_parser import ResourcesConfiguration
from ..Utils.io import load_nifti_volume
from .AbstractPipelineStep import AbstractPipelineStep


class ClassificationStep(AbstractPipelineStep):
    """
    The only use-case is an MRI sequence type classifier for now, to run on all inputs as a preprocessing step.
    """
    _model_name = None
    _patient_parameters = None
    _working_folder = None
    _input_volume_uid = None
    _input_volume_filepath = None

    def __init__(self, step_json: dict):
        super(ClassificationStep, self).__init__(step_json=step_json)
        self.__reset()
        self._model_name = self._step_json["model"]

    def __reset(self):
        self._model_name = None
        self._patient_parameters = None
        self._working_folder = None
        self._input_volume_uid = None
        self._input_volume_filepath = None

    def setup(self, patient_parameters):
        self._patient_parameters = patient_parameters

        self._working_folder = os.path.join(ResourcesConfiguration.getInstance().output_folder, "classification_tmp")
        os.makedirs(self._working_folder, exist_ok=True)
        os.makedirs(os.path.join(self._working_folder, 'inputs'), exist_ok=True)
        os.makedirs(os.path.join(self._working_folder, 'outputs'), exist_ok=True)

    def execute(self):
        try:
            if len(self._step_json["inputs"].keys()) == 0:
                for volume_uid in self._patient_parameters.get_all_radiological_volume_uids():
                    self._input_volume_uid = volume_uid
                    self._input_volume_filepath = self._patient_parameters.get_radiological_volume(volume_uid=volume_uid).get_usable_input_filepath()
                    new_fp = os.path.join(self._working_folder, 'inputs', 'input0.nii.gz')
                    shutil.copyfile(self._input_volume_filepath, new_fp)
                    self.__perform_classification()
            else:  # Not a use-case for the moment.
                pass

            if os.path.exists(self._working_folder):
                shutil.rmtree(self._working_folder)

            # Dumping the classification results for reloading afterwards
            # Only/current use-case for now: MRI sequence classification.
            classification_results_filename = os.path.join(ResourcesConfiguration.getInstance().output_folder, "mri_sequences.csv")
            classes = []
            for volume_uid in self._patient_parameters.get_all_radiological_volume_uids():
                classes.append([os.path.basename(self._patient_parameters.get_radiological_volume(volume_uid).get_raw_input_filepath()),
                                self._patient_parameters.get_radiological_volume(volume_uid).get_sequence_type_str()])
            df = pd.DataFrame(classes, columns=['File', 'MRI sequence'])
            df.to_csv(classification_results_filename, index=False)
            logging.info("Classification results written to {}".format(classification_results_filename))
        except Exception as e:
            logging.error("[ClassificationStep] Automatic classification failed with: {}".format(traceback.format_exc()))
            raise ValueError("[ClassificationStep] Automatic classification failed.")

        return self._patient_parameters

    def __perform_classification(self) -> None:
        """
        @TODO. Should hold the sequence class for each input, and dump a report/summary for Raidionics/the user.
        """
        try:
            tmp_dir = os.path.join(ResourcesConfiguration.getInstance().output_folder, "classification_tmp")
            os.makedirs(tmp_dir, exist_ok=True)
            classification_config = configparser.ConfigParser()
            classification_config.add_section('System')
            classification_config.set('System', 'gpu_id', ResourcesConfiguration.getInstance().gpu_id)
            classification_config.set('System', 'inputs_folder', os.path.join(self._working_folder, 'inputs'))
            classification_config.set('System', 'output_folder', os.path.join(self._working_folder, 'outputs'))
            classification_config.set('System', 'model_folder',
                                      os.path.join(ResourcesConfiguration.getInstance().model_folder, self._model_name))
            classification_config.add_section('Runtime')
            classification_config.set('Runtime', 'reconstruction_method', 'thresholding')
            classification_config.set('Runtime', 'reconstruction_order', 'resample_first')
            classification_config.add_section('Neuro')
            classification_config.set('Neuro', 'brain_segmentation_filename', '')
            classification_config_filename = os.path.join(os.path.join(self._working_folder, 'inputs'),
                                                          'classification_config.ini')
            with open(classification_config_filename, 'w') as outfile:
                classification_config.write(outfile)

            log_level = logging.getLogger().level
            log_str = 'warning'
            if log_level == 10:
                log_str = 'debug'
            elif log_level == 20:
                log_str = 'info'
            elif log_level == 40:
                log_str = 'error'

            from raidionicsseg.fit import run_model
            run_model(classification_config_filename)
        except Exception as e:
            logging.error("[ClassificationStep] Automatic classification failed with: {}".format(traceback.format_exc()))
            raise ValueError("[ClassificationStep] Automatic classification failed.")

        try:
            classification_results_filename = os.path.join(os.path.join(self._working_folder, 'outputs'),
                                                           'classification-results.csv')
            classification_results_df = pd.read_csv(classification_results_filename)
            final_class = classification_results_df.values[classification_results_df[classification_results_df.columns[1]].idxmax(), 0]
            self._patient_parameters.get_radiological_volume(volume_uid=self._input_volume_uid).set_sequence_type(final_class)
        except Exception as e:
            logging.error("[ClassificationStep] Classification results parsing failed with: {}".format(traceback.format_exc()))
            raise ValueError("[ClassificationStep] Classification results parsing failed.")

        return
