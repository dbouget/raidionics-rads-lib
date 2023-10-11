import os
import shutil
import numpy as np
import nibabel as nib
import logging
import configparser
import traceback
from ..Utils.utilities import get_type_from_string, get_type_from_enum_name
from ..Utils.configuration_parser import ResourcesConfiguration
from ..Utils.io import load_nifti_volume
from .AbstractPipelineStep import AbstractPipelineStep
from ..Utils.DataStructures.PatientStructure import PatientParameters
from ..Utils.DataStructures.AnnotationStructure import Annotation, AnnotationClassType, BrainTumorType


class SegmentationStep(AbstractPipelineStep):
    """
    Calling the raidionics_seg_lib backend, after generating the proper inputs and filling in the runtime config.ini.
    """
    _input_volume_uid = None  # Internal unique id to the first/main radiological volume.
    _segmentation_targets = None  # String(s) matching elements of AnnotationClassType, multiple annotations can be returned by a single model.
    _segmentation_output_type = None  # String describing how the predictions should be stored, either raw or thresholded.
    _model_name = None  # Basename of the folder containing the model to execute
    _patient_parameters = None  # Overall patient parameters, updated on-the-fly
    _working_folder = None  # Temporary directory on disk to store inputs/outputs for the segmentation

    def __init__(self, step_json: dict):
        super(SegmentationStep, self).__init__(step_json=step_json)
        self.__reset()
        # @TODO. Extend the model_name if multiple are available based on the number of inputs (postop seg) with e.g. _1c, _2c, etc...
        self._model_name = self._step_json["model"]
        self._segmentation_targets = self._step_json["target"]
        self._segmentation_output_type = None
        if "format" in self._step_json:
            self._segmentation_output_type = self._step_json["format"]

    def __reset(self):
        self._input_volume_uid = None
        self._segmentation_targets = None
        self._segmentation_output_type = None
        self._model_name = None
        self._patient_parameters = None
        self._working_folder = None

    def setup(self, patient_parameters: PatientParameters) -> None:
        """
        Sanity check that all requirements are met for running the segmentation step. Preparation of all data inputs.

        Parameters
        ----------
        patient_parameters: PatientParameters
            Placeholder for the current patient data, which will be updated with the results of this step.
        """
        self._patient_parameters = patient_parameters

        self._working_folder = os.path.join(ResourcesConfiguration.getInstance().output_folder, "segmentation_tmp")
        os.makedirs(self._working_folder, exist_ok=True)
        os.makedirs(os.path.join(self._working_folder, 'inputs'), exist_ok=True)
        os.makedirs(os.path.join(self._working_folder, 'outputs'), exist_ok=True)

        try:
            for k in list(self._step_json["inputs"].keys()):
                input_json = self._step_json["inputs"][k]
                # Use-case where the radiological volume should be used in its original reference space
                if input_json["space"]["timestamp"] == input_json["timestamp"] and \
                        input_json["space"]["sequence"] == input_json["sequence"]:
                    volume_uid = self._patient_parameters.get_radiological_volume_uid(timestamp=input_json["timestamp"],
                                                                                      sequence=input_json["sequence"])
                    if volume_uid == "-1":
                        raise ValueError("No radiological volume for {}.".format(input_json))

                    # Assuming the first input is actually the final target. Might need to add another parameter
                    # for specifying the volume the annotation is linked to, if multiple inputs.
                    if not self._input_volume_uid:
                        self._input_volume_uid = volume_uid
                    # Use-case where the input is actually an annotation and not a raw radiological volume
                    if input_json["labels"]:
                        annotation_type = get_type_from_enum_name(AnnotationClassType, input_json["labels"])
                        anno_uids = self._patient_parameters.get_all_annotations_uids_class_radiological_volume(volume_uid=volume_uid,
                                                                                                                annotation_class=annotation_type)
                        if len(anno_uids) == 0:
                            raise ValueError("No annotation for {}.".format(input_json))
                        anno_uid = anno_uids[0]
                        input_fp = self._patient_parameters.get_annotation(annotation_uid=anno_uid).get_usable_input_filepath()
                        if not os.path.exists(input_fp):
                            raise ValueError("No annotation file on disk for {}.".format(input_fp))
                        new_fp = os.path.join(self._working_folder, 'inputs', 'input' + str(k) + '.nii.gz')
                        shutil.copyfile(input_fp, new_fp)
                    else:
                        if volume_uid != "-1":
                            input_fp = self._patient_parameters.get_radiological_volume(volume_uid=volume_uid).get_usable_input_filepath()
                            if not os.path.exists(input_fp):
                                raise ValueError("No radiological volume file on disk for {}.".format(input_fp))
                            new_fp = os.path.join(self._working_folder, 'inputs', 'input' + str(k) + '.nii.gz')
                            shutil.copyfile(input_fp, new_fp)
                        else:
                            raise ValueError("No radiological volume for {}.".format(input_json))

                # Use-case where the radiological volume should be used in another reference space
                else:
                    volume_uid = self._patient_parameters.get_radiological_volume_uid(timestamp=input_json["timestamp"],
                                                                                      sequence=input_json["sequence"])
                    if volume_uid == "-1":
                        raise ValueError("No radiological volume for {}.".format(input_json))

                    ref_space_uid = self._patient_parameters.get_radiological_volume_uid(timestamp=input_json["space"]["timestamp"],
                                                                                         sequence=input_json["space"]["sequence"])
                    if ref_space_uid == "-1" and input_json["space"]["timestamp"] != "-1":
                        raise ValueError("No radiological volume for {}.".format(input_json["space"]))
                    else:  # @TODO. The reference space is an atlas, have to make an extra-pass for this.
                        pass

                    # Use-case where the input is actually an annotation and not a radiological volume
                    if input_json["labels"]:
                        annotation_type = get_type_from_enum_name(AnnotationClassType, input_json["labels"])
                        if annotation_type == -1:
                            raise ValueError("No AnnotationClassType matching {}.".format(input_json["labels"]))

                        anno_uids = self._patient_parameters.get_all_annotations_uids_class_radiological_volume(volume_uid=volume_uid,
                                                                                                                annotation_class=annotation_type)
                        if len(anno_uids) == 0:
                            raise ValueError("No annotation for {}.".format(input_json))
                        anno_uid = anno_uids[0]
                        input_fp = self._patient_parameters.get_annotation(annotation_uid=anno_uid).get_registered_volume_info(ref_space_uid)["filepath"]
                        if not os.path.exists(input_fp):
                            raise ValueError("No registered annotation file on disk for {}.".format(input_fp))
                        new_fp = os.path.join(self._working_folder, 'inputs', 'input' + str(k) + '.nii.gz')
                        shutil.copyfile(input_fp, new_fp)
                    # Use-case where the provided inputs are already co-registered
                    elif ResourcesConfiguration.getInstance().predictions_use_registered_data:
                        input_fp = self._patient_parameters.get_radiological_volume(volume_uid=volume_uid).get_usable_input_filepath()
                        if not os.path.exists(input_fp):
                            raise ValueError("No radiological volume file on disk for {}.".format(input_fp))
                        new_fp = os.path.join(self._working_folder, 'inputs', 'input' + str(k) + '.nii.gz')
                        shutil.copyfile(input_fp, new_fp)
                    else:
                        reg_fp = self._patient_parameters.get_radiological_volume(volume_uid=volume_uid).get_registered_volume_info(ref_space_uid)["filepath"]
                        if not os.path.exists(reg_fp):
                            raise ValueError("No registered radiological file on disk for {}.".format(reg_fp))
                        new_fp = os.path.join(self._working_folder, 'inputs', 'input' + str(k) + '.nii.gz')
                        shutil.copyfile(reg_fp, new_fp)
        except Exception as e:
            logging.error("[SegmentationStep] setup failed with: {}.".format(traceback.format_exc()))
            if os.path.exists(self._working_folder):
                shutil.rmtree(self._working_folder)
            raise ValueError("[SegmentationStep] setup failed.")

    def execute(self) -> PatientParameters:
        """
        Executes the current step.

        Returns
        -------
        PatientParameters
            Updated placeholder with the results of the current step.
        """
        if self._input_volume_uid:
            if ResourcesConfiguration.getInstance().diagnosis_task == 'neuro_diagnosis':
                self.__perform_neuro_segmentation()
            else:
                self.__perform_mediastinum_segmentation()
        return self._patient_parameters

    def __perform_neuro_segmentation(self) -> None:
        """
        """
        try:
            existing_uid = self._patient_parameters.get_all_annotations_uids_class_radiological_volume(
                volume_uid=self._input_volume_uid,
                annotation_class=get_type_from_enum_name(AnnotationClassType, self._segmentation_targets[0]))
            if len(existing_uid) != 0:
                # An annotation object matching the request already exists, hence skipping the step.
                logging.info("[SegmentationStep] Automatic segmentation skipped, results already existing.")
                if os.path.exists(self._working_folder):
                    shutil.rmtree(self._working_folder)
                return

            seg_config = configparser.ConfigParser()
            seg_config.add_section('System')
            seg_config.set('System', 'gpu_id', ResourcesConfiguration.getInstance().gpu_id)
            seg_config.set('System', 'inputs_folder', os.path.join(self._working_folder, 'inputs'))
            seg_config.set('System', 'output_folder', os.path.join(self._working_folder, 'outputs'))
            seg_config.set('System', 'model_folder', os.path.join(ResourcesConfiguration.getInstance().model_folder,
                                                                  self._model_name))
            seg_config.add_section('Runtime')
            seg_config.set('Runtime', 'reconstruction_method', ResourcesConfiguration.getInstance().predictions_reconstruction_method)
            if self._segmentation_output_type:
                seg_config.set('Runtime', 'reconstruction_method', self._segmentation_output_type)
            seg_config.set('Runtime', 'reconstruction_order', ResourcesConfiguration.getInstance().predictions_reconstruction_order)
            seg_config.set('Runtime', 'use_preprocessed_data', str(ResourcesConfiguration.getInstance().predictions_use_stripped_data))

            # @TODO. Have to be slightly improved, but should be working for our use-cases for now.
            existing_brain_annotations = self._patient_parameters.get_all_annotations_uids_class_radiological_volume(volume_uid=self._input_volume_uid,
                                                                                                                     annotation_class=AnnotationClassType.Brain)
            if len(existing_brain_annotations) != 0:
                seg_config.add_section('Neuro')
                seg_config.set('Neuro', 'brain_segmentation_filename',
                               self._patient_parameters.get_annotation(annotation_uid=existing_brain_annotations[0]).get_usable_input_filepath())
            seg_config_filename = os.path.join(os.path.join(self._working_folder, 'inputs'), 'seg_config.ini')
            with open(seg_config_filename, 'w') as outfile:
                seg_config.write(outfile)

            log_level = logging.getLogger().level
            log_str = 'warning'
            if log_level == 10:
                log_str = 'debug'
            elif log_level == 20:
                log_str = 'info'
            elif log_level == 40:
                log_str = 'error'

            from raidionicsseg.fit import run_model
            run_model(seg_config_filename)
        except Exception as e:
            logging.error("[SegmentationStep] Automatic segmentation failed with: {}.".format(traceback.format_exc()))
            if os.path.exists(self._working_folder):
                shutil.rmtree(self._working_folder)
            raise ValueError("[SegmentationStep] Automatic segmentation failed.")

        try:
            # Collecting the results and associating them with the parent radiological volume.
            generated_segmentations = []
            for _, _, files in os.walk(os.path.join(self._working_folder, 'outputs')):
                for f in files:
                    if 'nii.gz' in f:
                        generated_segmentations.append(f)
                break

            for s in generated_segmentations:
                label_name = s.split('_')[1].split('.')[0]
                if label_name in self._segmentation_targets:
                    seg_filename = os.path.join(os.path.join(self._working_folder, 'outputs'), s)
                    final_seg_filename = os.path.join(self._patient_parameters.get_radiological_volume(volume_uid=self._input_volume_uid).get_output_folder(),
                                                      os.path.basename(self._patient_parameters.get_radiological_volume(volume_uid=self._input_volume_uid).get_raw_input_filepath()).split('.')[0] + '_annotation-' + label_name + '.nii.gz')
                    if not os.path.exists(seg_filename):
                        raise ValueError("Segmentation results file could not be found on disk at {}".format(seg_filename))
                    shutil.move(seg_filename, final_seg_filename)
                    non_available_uid = True
                    anno_uid = None
                    while non_available_uid:
                        anno_uid = 'A' + str(np.random.randint(0, 10000))
                        if anno_uid not in self._patient_parameters.get_all_annotations_uids():
                            non_available_uid = False
                    annotation = Annotation(uid=anno_uid, input_filename=final_seg_filename,
                                            output_folder=self._patient_parameters.get_radiological_volume(volume_uid=self._input_volume_uid).get_output_folder(),
                                            radiological_volume_uid=self._input_volume_uid, annotation_class=label_name)
                    if label_name == 'Tumor':
                        subtype = "Glioblastoma"
                        if 'Meningioma' in self._model_name:
                            subtype = "Meningioma"
                        elif 'LGGlioma' in self._model_name:
                            subtype = "Lower-grade glioma"
                        elif 'Metastasis' in self._model_name:
                            subtype = "Metastasis"
                        annotation.set_annotation_subtype(type=BrainTumorType, value=subtype)
                    self._patient_parameters.include_annotation(anno_uid, annotation)
                    logging.info("Saved segmentation results in {}".format(final_seg_filename))
        except Exception as e:
            logging.error("[SegmentationStep] Segmentation results parsing failed with: {}.".format(traceback.format_exc()))
            if os.path.exists(self._working_folder):
                shutil.rmtree(self._working_folder)
            raise ValueError("[SegmentationStep] Segmentation results parsing failed.")

        if os.path.exists(self._working_folder):
            shutil.rmtree(self._working_folder)

    def __perform_mediastinum_segmentation(self):
        """

        """
        try:
            # Only looking if the first segmentation target exists, which is not optimal.
            # @TODO. Should check if all targets are existing to decide whether to run the model or not.
            existing_uid = self._patient_parameters.get_all_annotations_uids_class_radiological_volume(
                volume_uid=self._input_volume_uid,
                annotation_class=get_type_from_enum_name(AnnotationClassType, self._segmentation_targets[0]))
            if len(existing_uid) != 0:
                # An annotation object matching the request already exists, hence skipping the step.
                logging.info("[SegmentationStep] Automatic segmentation skipped, results already existing.")
                if os.path.exists(self._working_folder):
                    shutil.rmtree(self._working_folder)
                return

            seg_config = configparser.ConfigParser()
            seg_config.add_section('System')
            seg_config.set('System', 'gpu_id', ResourcesConfiguration.getInstance().gpu_id)
            # seg_config.set('System', 'input_filename', self._input_volume_filepath)
            seg_config.set('System', 'inputs_folder', os.path.join(self._working_folder, 'inputs'))
            seg_config.set('System', 'output_folder', os.path.join(self._working_folder, 'outputs'))
            seg_config.set('System', 'model_folder', os.path.join(ResourcesConfiguration.getInstance().model_folder,
                                                                  self._model_name))
            seg_config.add_section('Runtime')
            seg_config.set('Runtime', 'reconstruction_method',
                           ResourcesConfiguration.getInstance().predictions_reconstruction_method)
            if self._segmentation_output_type:
                seg_config.set('Runtime', 'reconstruction_method', self._segmentation_output_type)
            seg_config.set('Runtime', 'reconstruction_order', ResourcesConfiguration.getInstance().predictions_reconstruction_order)
            seg_config.set('Runtime', 'use_preprocessed_data', "True" if ResourcesConfiguration.getInstance().predictions_use_stripped_data else "False")

            # @TODO. Have to be slightly improved, but should be working for our use-cases for now.
            existing_lungs_annotations = self._patient_parameters.get_all_annotations_uids_class_radiological_volume(
                volume_uid=self._input_volume_uid,
                annotation_class=AnnotationClassType.Lungs)
            if len(existing_lungs_annotations) != 0:
                seg_config.add_section('Mediastinum')
                seg_config.set('Mediastinum', 'lungs_segmentation_filename',
                               self._patient_parameters.get_annotation(
                                   annotation_uid=existing_lungs_annotations[0]).get_usable_input_filepath())
            seg_config_filename = os.path.join(os.path.join(self._working_folder, 'inputs'), 'seg_config.ini')
            with open(seg_config_filename, 'w') as outfile:
                seg_config.write(outfile)

            log_level = logging.getLogger().level
            log_str = 'warning'
            if log_level == 10:
                log_str = 'debug'
            elif log_level == 20:
                log_str = 'info'
            elif log_level == 40:
                log_str = 'error'

            from raidionicsseg.fit import run_model
            run_model(seg_config_filename)
        except Exception as e:
            logging.error("[SegmentationStep] Automatic segmentation failed with: {}.".format(traceback.format_exc()))
            if os.path.exists(self._working_folder):
                shutil.rmtree(self._working_folder)
            raise ValueError("[SegmentationStep] Automatic segmentation failed.")

        try:
            # Collecting the results and associating them with the parent radiological volume.
            generated_segmentations = []
            for _, _, files in os.walk(os.path.join(self._working_folder, 'outputs')):
                for f in files:
                    if 'nii.gz' in f:
                        generated_segmentations.append(f)
                break

            for s in generated_segmentations:
                label_name = s.split('_')[1].split('.')[0]
                if label_name in self._segmentation_targets:
                    seg_filename = os.path.join(os.path.join(self._working_folder, 'outputs'), s)
                    final_seg_filename = os.path.join(self._patient_parameters.get_radiological_volume(
                        volume_uid=self._input_volume_uid).get_output_folder(),
                                                      os.path.basename(self._patient_parameters.get_radiological_volume(
                                                          volume_uid=self._input_volume_uid).get_raw_input_filepath()).split(
                                                          '.')[0] + '_annotation-' + label_name + '.nii.gz')
                    if not os.path.exists(seg_filename):
                        raise ValueError(
                            "Segmentation results file could not be found on disk at {}".format(seg_filename))
                    shutil.move(seg_filename, final_seg_filename)
                    non_available_uid = True
                    anno_uid = None
                    while non_available_uid:
                        anno_uid = 'A' + str(np.random.randint(0, 10000))
                        if anno_uid not in self._patient_parameters.get_all_annotations_uids():
                            non_available_uid = False
                    annotation = Annotation(uid=anno_uid, input_filename=final_seg_filename,
                                            output_folder=self._patient_parameters.get_radiological_volume(
                                                volume_uid=self._input_volume_uid).get_output_folder(),
                                            radiological_volume_uid=self._input_volume_uid, annotation_class=label_name)
                    self._patient_parameters.include_annotation(anno_uid, annotation)
                    logging.info("Saved segmentation results in {}".format(final_seg_filename))
        except Exception as e:
            logging.error(
                "[SegmentationStep] Segmentation results parsing failed with: {}.".format(traceback.format_exc()))
            if os.path.exists(self._working_folder):
                shutil.rmtree(self._working_folder)
            raise ValueError("[SegmentationStep] Segmentation results parsing failed.")

        if os.path.exists(self._working_folder):
            shutil.rmtree(self._working_folder)
