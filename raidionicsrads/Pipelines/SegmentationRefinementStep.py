import os
import shutil
import numpy as np
import nibabel as nib
import logging
import configparser
import traceback
from ..Utils.utilities import get_type_from_string, get_type_from_enum_name
from ..Utils.configuration_parser import ResourcesConfiguration
from ..Utils.volume_utilities import prediction_binary_dilation
from ..Utils.io import load_nifti_volume
from .AbstractPipelineStep import AbstractPipelineStep
from ..Utils.DataStructures.PatientStructure import PatientParameters
from ..Utils.DataStructures.AnnotationStructure import Annotation, AnnotationClassType, BrainTumorType


class SegmentationRefinementStep(AbstractPipelineStep):
    """
    Processing the results from the raidionics_seg_lib backend to improve or clean the predictions
    """
    _input_volume_uid = None  # Internal unique id to the first/main radiological volume.
    _patient_parameters = None  # Overall patient parameters, updated on-the-fly
    _working_folder = None  # Temporary directory on disk to store inputs/outputs for the segmentation
    _refinement_operation = None  # Type of refinement to operate
    _refinement_args = None  # Generic arguments needed for the specified refinement operation

    def __init__(self, step_json: dict):
        super(SegmentationRefinementStep, self).__init__(step_json=step_json)
        self.__reset()
        self._refinement_operation = self._step_json["operation"]
        self._refinement_args = self._step_json["args"]

    def __reset(self):
        self._input_volume_uid = None
        self._patient_parameters = None
        self._working_folder = None
        self._refinement_operation = None
        self._refinement_args = None

    def setup(self, patient_parameters: PatientParameters) -> None:
        """
        Sanity check that all requirements are met for running the segmentation step. Preparation of all data inputs.

        Parameters
        ----------
        patient_parameters: PatientParameters
            Placeholder for the current patient data, which will be updated with the results of this step.
        """
        self._patient_parameters = patient_parameters

        self._working_folder = os.path.join(ResourcesConfiguration.getInstance().output_folder, "seg_refinement_tmp")
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
                        new_fp = os.path.join(self._working_folder, 'inputs', 'input' + str(k) + '_' + str(annotation_type) + '.nii.gz')
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
                        new_fp = os.path.join(self._working_folder, 'inputs', 'input' + str(k) + '_' + str(annotation_type) + '.nii.gz')
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
            logging.error("[SegmentationRefinementStep] setup failed with: {}.".format(traceback.format_exc()))
            if os.path.exists(self._working_folder):
                shutil.rmtree(self._working_folder)
            raise ValueError("[SegmentationRefinementStep] setup failed.")

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
            segmentation_folder = os.path.join(self._working_folder, 'inputs')
            segmentation_files = []
            for _, _, files in os.walk(segmentation_folder):
                for f in files:
                    segmentation_files.append(f)
                break

            for fn in segmentation_files:
                seg_fn = os.path.join(segmentation_folder, fn)
                seg_ni = nib.load(seg_fn)
                seg = seg_ni.get_fdata()[:]

                res = None
                if self._refinement_operation == "dilation":
                    res = prediction_binary_dilation(seg.astype('uint8'),
                                                     voxel_volume=np.prod(seg_ni.header.get_zooms()) * 1e-3,
                                                     arg=int(self._refinement_args))
                else:
                    raise ValueError("[RefinementStep] The selected refinement operation is not available, with value {}".format(self._refinement_operation))

                res_ni = nib.Nifti1Image(res, affine=seg_ni.affine)
                output_fn = os.path.join(self._working_folder, 'outputs', fn)
                nib.save(res_ni, output_fn)

        except Exception as e:
            logging.error("[SegmentationRefinementStep] Segmentation refinement failed with: {}.".format(traceback.format_exc()))
            if os.path.exists(self._working_folder):
                shutil.rmtree(self._working_folder)
            raise ValueError("[SegmentationRefinementStep] Segmentation refinement failed.")

        try:
            # Collecting the results and associating them with the parent radiological volume.
            generated_segmentations = []
            for _, _, files in os.walk(os.path.join(self._working_folder, 'outputs')):
                for f in files:
                    if 'nii.gz' in f:
                        generated_segmentations.append(f)
                break

            for k in list(self._step_json["inputs"].keys()):
                output_index = ["input" + str(k) in x for x in generated_segmentations].index(True)
                seg_filename = os.path.join(self._working_folder, "outputs", generated_segmentations[output_index])
                annotation_struct = self._patient_parameters.get_input_from_json(self._step_json["inputs"][k])
                shutil.copyfile(src=seg_filename, dst=annotation_struct.raw_input_filepath)

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
            raise ValueError("NIY")
        except Exception as e:
            logging.error(
                "[SegmentationRefinementStep] Process failed with: {}.".format(traceback.format_exc()))
            raise ValueError("[SegmentationRefinementStep] Process failed.")
