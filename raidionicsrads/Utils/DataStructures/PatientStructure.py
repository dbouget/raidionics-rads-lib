import os
import numpy as np
import re
import pandas as pd
import logging
from typing import List
from ..configuration_parser import ResourcesConfiguration
from ..utilities import input_file_category_disambiguation, get_type_from_enum_name
from .RadiologicalVolumeStructure import RadiologicalVolume
from .AnnotationStructure import Annotation, AnnotationClassType
from .RegistrationStructure import Registration


class PatientParameters:
    _unique_id = None  # Internal unique identifier for the patient
    _input_filepath = None  # Folder path containing all the data for the current patient.
    _timestamps = {}  # All timestamps for the current patient.
    _radiological_volumes = {}  # All radiological volume instances loaded for the current patient.
    _annotation_volumes = {}  # All Annotation instances loaded for the current patient.
    _atlas_volumes = {}  # All Atlas instances loaded for the current patient.
    _registrations = {}  # All registration transforms.
    _reportings = {}  # All clinical reports (if applicable).

    def __init__(self, id: str, patient_filepath: str):
        """
        """
        self.__reset()
        self._unique_id = id
        self._input_filepath = patient_filepath

        if not patient_filepath or not os.path.exists(patient_filepath):
            # Error case
            return

        self.__init_from_scratch()

    def __reset(self):
        """
        All objects share class or static variables.
        An instance or non-static variables are different for different objects (every object has a copy).
        """
        self._unique_id = None
        self._input_filepath = None
        self._timestamps = {}
        self._radiological_volumes = {}
        self._annotation_volumes = {}
        self._atlas_volumes = {}
        self._registrations = {}
        self._reportings = {}

    def __init_from_scratch(self):
        """
        Iterating through the patient folder to identify the radiological volumes for each timestamp.
        """
        timestamp_folders = []
        for _, dirs, _ in os.walk(self._input_filepath):
            for d in dirs:
                timestamp_folders.append(d)
            break

        ts_folders_dict = {}
        for i in timestamp_folders:
            if re.search(r'\d+', i):  # Skipping folders without an integer inside, otherwise assuming timestamps from 0 onwards
                ts_folders_dict[int(re.search(r'\d+', i).group())] = i

        ordered_ts_folders = dict(sorted(ts_folders_dict.items(), key=lambda item: item[0], reverse=False))

        for i, ts in enumerate(list(ordered_ts_folders.keys())):
            ts_folder = os.path.join(self._input_filepath, ordered_ts_folders[ts])
            if ResourcesConfiguration.getInstance().caller == 'raidionics':  # Specifics to cater to Raidionics
                ts_folder = os.path.join(ts_folder, 'raw')
            patient_files = []

            timestamp_uid = "T" + str(i)
            timestamp_instance = TimestampParameters(id=timestamp_uid, timestamp_filepath=ts_folder)
            self._timestamps[timestamp_uid] = timestamp_instance

            for _, _, files in os.walk(ts_folder):
                for f in files:
                    if '.'.join(f.split('.')[1:]) in ResourcesConfiguration.getInstance().get_accepted_image_formats():
                        patient_files.append(f)
                break

            annotation_files = []
            for f in patient_files:
                file_content_type = input_file_category_disambiguation(os.path.join(ts_folder, f))
                # Generating a unique id for the radiological volume
                if file_content_type == "Volume":
                    base_data_uid = os.path.basename(f).strip().split('.')[0]
                    non_available_uid = True
                    while non_available_uid:
                        data_uid = 'V' + str(np.random.randint(0, 10000)) + '_' + base_data_uid
                        if data_uid not in list(self._radiological_volumes.keys()):
                            non_available_uid = False
                    self._radiological_volumes[data_uid] = RadiologicalVolume(uid=data_uid,
                                                                              input_filename=os.path.join(ts_folder, f),
                                                                              timestamp_uid=timestamp_uid)
                elif file_content_type == "Annotation":
                    annotation_files.append(f)

            # Iterating over the annotation files in a second time, when all the parent objects have been created
            for f in annotation_files:
                # Collecting the base name of the radiological volume, often before a label or annotation tag
                base_name = os.path.basename(f).strip().split('.')[0].split('label')[0][:-1]
                if ResourcesConfiguration.getInstance().caller == 'raidionics':
                    base_name = os.path.basename(f).strip().split('.')[0].split('annotation')[0][:-1]
                parent_link = [base_name in x for x in list(self._radiological_volumes.keys())]
                if True in parent_link:
                    parent_uid = list(self._radiological_volumes.keys())[parent_link.index(True)]
                    non_available_uid = True
                    while non_available_uid:
                        data_uid = 'A' + str(np.random.randint(0, 10000)) + '_' + base_name
                        if data_uid not in list(self._annotation_volumes.keys()):
                            non_available_uid = False
                    if ResourcesConfiguration.getInstance().caller == 'raidionics':
                        class_name = os.path.basename(f).strip().split('.')[0].split('annotation')[1][1:]
                    else:
                        class_name = os.path.basename(f).strip().split('.')[0].split('label')[1][1:]
                    self._annotation_volumes[data_uid] = Annotation(uid=data_uid,
                                                                    input_filename=os.path.join(ts_folder, f),
                                                                    output_folder=self._radiological_volumes[parent_uid].get_output_folder(),
                                                                    radiological_volume_uid=parent_uid,
                                                                    annotation_class=class_name)
                else:
                    # Case where the annotation does not match any radiological volume, has to be left aside
                    pass
        sequences_filename = os.path.join(self._input_filepath, 'mri_sequences.csv')
        if os.path.exists(sequences_filename):
            df = pd.read_csv(sequences_filename)
            volume_basenames = list(df['File'].values)
            for vn in volume_basenames:
                volume_object = self.get_radiological_volume_by_base_filename(vn)
                if volume_object:
                    volume_object.set_sequence_type(df.loc[df['File'] == vn]['MRI sequence'].values[0])
                else:
                    logging.warning("[PatientStructure] Filename {} not matching any radiological volume volume.".format(vn))

    def include_annotation(self, anno_uid, annotation):
        self._annotation_volumes[anno_uid] = annotation

    def include_registration(self, reg_uid, registration):
        self._registrations[reg_uid] = registration

    def include_reporting(self, report_uid, report):
        self._reportings[report_uid] = report

    def get_input_from_json(self, input_json: dict):
        """
        Automatic identifies and returns the proper structure instance based on the content of the input json dict.
        """
        # Use-case where the input is in its original reference space
        if input_json["space"]["timestamp"] == input_json["timestamp"] and \
                input_json["space"]["sequence"] == input_json["sequence"]:
            volume_uid = self.get_radiological_volume_uid(timestamp=input_json["timestamp"],
                                                          sequence=input_json["sequence"])
            # Use-case where the input is actually an annotation and not a raw radiological volume
            if input_json["labels"]:
                annotation_type = get_type_from_enum_name(AnnotationClassType, input_json["labels"])
                anno_uids = self.get_all_annotations_uids_class_radiological_volume(volume_uid=volume_uid,
                                                                                    annotation_class=annotation_type)
                if len(anno_uids) != 0:
                    return self.get_annotation(anno_uids[0])
                else:
                    raise ValueError("No annotation file existing for the specified json input with:\n {}.\n".format(input_json))
            else:
                return self.get_radiological_volume(volume_uid)
        else:  # The input is in a registered space
            volume_uid = self.get_radiological_volume_uid(timestamp=input_json["timestamp"],
                                                          sequence=input_json["sequence"])
            if volume_uid == "-1":
                raise ValueError("No radiological volume for {}.".format(input_json))

            ref_space_uid = self.get_radiological_volume_uid(timestamp=input_json["space"]["timestamp"],
                                                             sequence=input_json["space"]["sequence"])
            if ref_space_uid == "-1" and input_json["space"]["timestamp"] != "-1":
                raise ValueError("No radiological volume for {}.".format(input_json))
            else:  # @TODO. The reference space is an atlas, have to make an extra-pass for this.
                pass
            # Use-case where the input is actually an annotation and not a raw radiological volume
            if input_json["labels"]:
                annotation_type = get_type_from_enum_name(AnnotationClassType, input_json["labels"])
                if annotation_type == -1:
                    raise ValueError("No radiological volume for {}.".format(input_json))

                anno_uids = self.get_all_annotations_uids_class_radiological_volume(volume_uid=volume_uid,
                                                                                    annotation_class=annotation_type)
                if len(anno_uids) == 0:
                    raise ValueError("No radiological volume for {}.".format(input_json))
                anno_uid = anno_uids[0]
                volume = self.get_annotation(annotation_uid=anno_uid).get_registered_volume_info(ref_space_uid)
                input_fp = volume["filepath"]
                if not os.path.exists(input_fp):
                    raise ValueError("No radiological volume for {}.".format(input_json))
                else:
                    return volume
                # Use-case where the provided inputs are already co-registered
            elif ResourcesConfiguration.getInstance().predictions_use_registered_data:
                volume = self.get_radiological_volume(volume_uid=volume_uid)
                input_fp = volume.get_usable_input_filepath()
                if not os.path.exists(input_fp):
                    raise ValueError("No radiological volume for {}.".format(input_json))
                else:
                    return volume
            else:
                volume = self.get_radiological_volume(volume_uid=volume_uid).get_registered_volume_info(ref_space_uid)
                reg_fp = volume["filepath"]
                if not os.path.exists(reg_fp):
                    raise ValueError("No radiological volume for {}.".format(input_json))
                else:
                    return volume

    def get_all_radiological_volume_uids(self) -> List[str]:
        return list(self._radiological_volumes.keys())

    def get_radiological_volume_uid(self, timestamp: int, sequence: str) -> str:
        for v in self._radiological_volumes.keys():
            if self._radiological_volumes[v]._timestamp_id == "T" + str(timestamp) and str(self._radiological_volumes[v]._sequence_type) == sequence:
                return v
        return "-1"

    def get_radiological_volume(self, volume_uid: str) -> RadiologicalVolume:
        return self._radiological_volumes[volume_uid]

    def get_radiological_volume_by_base_filename(self, base_fn: str):
        result = None
        for im in self._radiological_volumes:
            if os.path.basename(self._radiological_volumes[im].get_usable_input_filepath()) == base_fn:
                return self._radiological_volumes[im]
        return result

    def get_all_annotations_uids(self) -> List[str]:
        return list(self._annotation_volumes.keys())

    def get_annotation(self, annotation_uid: str) -> Annotation:
        return self._annotation_volumes[annotation_uid]

    def get_all_annotations_uids_radiological_volume(self, volume_uid: str) -> List[str]:
        res = []
        for v in self._annotation_volumes.keys():
            if self._annotation_volumes[v]._radiological_volume_uid == volume_uid:
                res.append(v)
        return res

    def get_all_annotations_uids_class_radiological_volume(self, volume_uid: str,
                                                           annotation_class: AnnotationClassType) -> List[str]:
        res = []
        for v in self._annotation_volumes.keys():
            if self._annotation_volumes[v]._radiological_volume_uid == volume_uid and \
                    self._annotation_volumes[v]._annotation_type == annotation_class:
                res.append(v)
        return res

    def get_registration_by_uids(self, fixed_uid: str, moving_uid: str) -> Registration:
        registration = None
        for r in list(self._registrations.keys()):
            if self._registrations[r]._fixed_uid == fixed_uid and self._registrations[r]._moving_uid == moving_uid:
                return self._registrations[r]
        return registration

    def get_registration_by_json(self, fixed: dict, moving: dict) -> Registration:
        fixed_ts = fixed["timestamp"]
        fixed_seq = fixed["sequence"]

        moving_ts = moving["timestamp"]
        moving_seq = moving["sequence"]

        fixed_uid = None
        moving_uid = None
        if fixed_ts == -1:
            fixed_uid = 'MNI'
        else:
            fixed_uid = self.get_radiological_volume_uid(fixed_ts, fixed_seq)

        if moving_ts == -1:
            moving_uid = 'MNI'
        else:
            moving_uid = self.get_radiological_volume_uid(moving_ts, moving_seq)

        return self.get_registration_by_uids(fixed_uid=fixed_uid, moving_uid=moving_uid)

    def get_all_reportings_uids(self) -> List[str]:
        return list(self._reportings.keys())


class TimestampParameters:
    _unique_id = None  # Internal unique identifier for the patient
    _input_filepath = None  #

    def __init__(self, id: str, timestamp_filepath: str):
        """
        """
        self.__reset()
        self._unique_id = id
        self._input_filepath = timestamp_filepath

        self.__init_from_scratch()

    def __reset(self):
        """
        All objects share class or static variables.
        An instance or non-static variables are different for different objects (every object has a copy).
        """
        self._unique_id = None
        self._input_filepath = None

    def __init_from_scratch(self):
        pass
