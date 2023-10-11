import os
import shutil
import numpy as np
import nibabel as nib
import logging
import configparser
import traceback
from tqdm import tqdm
from ..Utils.configuration_parser import ResourcesConfiguration
from ..Utils.io import load_nifti_volume
from ..Utils.ants_registration import *
from ..Processing.brain_processing import *
from .AbstractPipelineStep import AbstractPipelineStep
from ..Utils.DataStructures.AnnotationStructure import AnnotationClassType
from ..Utils.DataStructures.RegistrationStructure import Registration


class RegistrationDeployerStep(AbstractPipelineStep):
    _patient_parameters = None
    _moving_volume_uid = None
    _fixed_volume_uid = None
    _registration_runner = None
    _registration_instance = None
    _direction = None

    def __init__(self, step_json: dict):
        super(RegistrationDeployerStep, self).__init__(step_json=step_json)
        self.__reset()
        self._registration_runner = ANTsRegistration()
        self._direction = self._step_json["direction"]

    def __reset(self):
        self._patient_parameters = None
        self._moving_volume_uid = None
        self._fixed_volume_uid = None
        self._registration_runner = None
        self._registration_instance = None
        self._direction = None

    def setup(self, patient_parameters):
        """
        """
        self._patient_parameters = patient_parameters
        self._registration_instance = self._patient_parameters.get_registration_by_json(fixed=self._step_json["fixed"],
                                                                                  moving=self._step_json["moving"])
        self._fixed_volume_uid = self._registration_instance._fixed_uid
        self._moving_volume_uid = self._registration_instance._moving_uid
        self._registration_runner.reg_transform['fwdtransforms'] = self._registration_instance._forward_filepaths
        self._registration_runner.reg_transform['invtransforms'] = self._registration_instance._inverse_filepaths

    def execute(self):
        """
        @TODO. Have to finish/improve how to deal with atlases, can't be the same process as regular volumes
        @TODO2. Have to take into account the registration direction from [forward, inverse].
        @TODO3. Have to deal with neuro/mediastinum, or more use-cases
        """
        if self._moving_volume_uid != 'MNI' and self._direction == 'forward':
            self.__apply_registration()
            self.__apply_registration_annotations()
        elif self._moving_volume_uid != 'MNI' and self._direction == 'inverse':
            self.__apply_registration_atlas_space()

        self._registration_runner.clear_output_folder()
        return self._patient_parameters

    def __apply_registration(self):
        fixed_filepath = None
        if self._fixed_volume_uid == 'MNI':
            fixed_filepath = ResourcesConfiguration.getInstance().mni_atlas_filepath_T1
        else:
            fixed_filepath = self._patient_parameters.get_radiological_volume(volume_uid=self._fixed_volume_uid).get_usable_input_filepath()

        moving_filepath = None
        if self._moving_volume_uid == 'MNI':
            moving_filepath = ResourcesConfiguration.getInstance().mni_atlas_filepath_T1
        else:
            moving_filepath = self._patient_parameters.get_radiological_volume(volume_uid=self._moving_volume_uid).get_usable_input_filepath()

        fp = self._registration_runner.apply_registration_transform(moving=moving_filepath, fixed=fixed_filepath,
                                                                    interpolation='bSpline')
        new_fp = os.path.join(self._patient_parameters.get_radiological_volume(volume_uid=self._moving_volume_uid).get_output_folder(),
                              self._fixed_volume_uid + '_space',
                              self._moving_volume_uid + '_registered_to_' + self._fixed_volume_uid + '.nii.gz')
        os.makedirs(os.path.dirname(new_fp), exist_ok=True)
        shutil.copyfile(fp, new_fp)
        self._patient_parameters.get_radiological_volume(volume_uid=self._moving_volume_uid).include_registered_volume(filepath=new_fp,
                                                                                                                       registration_uid=self._registration_instance.get_unique_id(),
                                                                                                                       destination_space_uid=self._fixed_volume_uid)

    def __apply_registration_annotations(self):
        fixed_filepath = None
        if self._fixed_volume_uid == 'MNI':
            fixed_filepath = ResourcesConfiguration.getInstance().mni_atlas_filepath_T1
        else:
            fixed_filepath = self._patient_parameters.get_radiological_volume(volume_uid=self._fixed_volume_uid).get_usable_input_filepath()

        for anno in self._patient_parameters.get_all_annotations_uids_radiological_volume(volume_uid=self._moving_volume_uid):
            annotation = self._patient_parameters.get_annotation(annotation_uid=anno)
            moving_filepath = annotation.get_usable_input_filepath()

            fp = self._registration_runner.apply_registration_transform(moving=moving_filepath, fixed=fixed_filepath,
                                                                        interpolation='nearestNeighbor')
            new_fp = os.path.join(self._patient_parameters.get_radiological_volume(volume_uid=self._moving_volume_uid).get_output_folder(),
                                  self._fixed_volume_uid + '_space',
                                  self._moving_volume_uid + '_label_' + annotation.get_annotation_type_str() + '_registered_to_' + self._fixed_volume_uid + '.nii.gz')
            os.makedirs(os.path.dirname(new_fp), exist_ok=True)
            shutil.copyfile(fp, new_fp)
            annotation.include_registered_volume(filepath=new_fp,
                                                 registration_uid=self._registration_instance.get_unique_id(),
                                                 destination_space_uid=self._fixed_volume_uid)

    def __apply_registration_atlas_space(self):
        """
        @TODO. Have to include this info somehow inside the self._patient_parameters
        """
        # fixed_filepath = ResourcesConfiguration.getInstance().mni_atlas_filepath_T1
        fixed_filepath = self._patient_parameters.get_radiological_volume(volume_uid=self._moving_volume_uid).get_usable_input_filepath()
        dump_folder = os.path.join(self._patient_parameters.get_radiological_volume(volume_uid=self._moving_volume_uid).get_output_folder(),
                                   'Cortical-structures')
        os.makedirs(dump_folder, exist_ok=True)

        for s in ResourcesConfiguration.getInstance().neuro_features_cortical_structures:
            fp = self._registration_runner.apply_registration_inverse_transform(
                moving=ResourcesConfiguration.getInstance().cortical_structures['MNI'][s]['Mask'],
                fixed=fixed_filepath, interpolation='nearestNeighbor', label='Cortical-structures/' + s)

            new_fp = os.path.join(dump_folder, self._fixed_volume_uid + '_' + s + '_atlas.nii.gz')
            shutil.copyfile(fp, new_fp)

        bcb_tracts_cutoff = 0.5
        dump_folder = os.path.join(self._patient_parameters.get_radiological_volume(volume_uid=self._moving_volume_uid).get_output_folder(),
                                   'Subcortical-structures')
        os.makedirs(dump_folder, exist_ok=True)

        for s in ResourcesConfiguration.getInstance().neuro_features_subcortical_structures:
            for i, elem in enumerate(tqdm(ResourcesConfiguration.getInstance().subcortical_structures['MNI'][s]['Singular'].keys())):
                raw_filename = ResourcesConfiguration.getInstance().subcortical_structures['MNI'][s]['Singular'][elem]
                raw_tract_ni = nib.load(raw_filename)
                raw_tract = raw_tract_ni.get_fdata()[:]
                raw_tract[raw_tract < bcb_tracts_cutoff] = 0
                raw_tract[raw_tract >= bcb_tracts_cutoff] = 1
                raw_tract = raw_tract.astype('uint8')
                dump_filename = os.path.join(self._registration_runner.registration_folder, os.path.basename(raw_filename))
                os.makedirs(os.path.dirname(dump_filename), exist_ok=True)
                nib.save(nib.Nifti1Image(raw_tract, affine=raw_tract_ni.affine), dump_filename)

                fp = self._registration_runner.apply_registration_inverse_transform(
                    moving=dump_filename,
                    fixed=fixed_filepath,
                    interpolation='nearestNeighbor',
                    label='Subcortical-structures/' + os.path.basename(raw_filename).split('.')[0].replace('_mni', ''))
                new_fp = os.path.join(dump_folder, self._fixed_volume_uid + '_' + s + '_atlas' + elem + '.nii.gz')
                shutil.copyfile(fp, new_fp)

            overall_mask_filename = ResourcesConfiguration.getInstance().subcortical_structures['MNI'][s]['Mask']
            fp = self._registration_runner.apply_registration_inverse_transform(
                moving=overall_mask_filename,
                fixed=fixed_filepath,
                interpolation='nearestNeighbor',
                label='Subcortical-structures/' + s)

            new_fp = os.path.join(dump_folder, self._fixed_volume_uid + '_' + s + '_atlas.nii.gz')
            shutil.copyfile(fp, new_fp)
