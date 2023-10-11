import os
from typing import List
from aenum import Enum, unique
from ..utilities import get_type_from_string, input_file_type_conversion
from ..configuration_parser import ResourcesConfiguration

@unique
class RadiologicalType(Enum):
    """

    """
    _init_ = 'value string'

    MRI = 0, 'MRI'  # MRI Series
    CT = 1, 'CT'  # Scan CT

    def __str__(self):
        return self.string


@unique
class MRISequenceType(Enum):
    """

    """
    _init_ = 'value string'

    T1w = 0, 'T1-w'  # T1-weighted sequence
    T1c = 1, 'T1-CE'  # Gd-enhanced T1-weighted sequence
    T2 = 2, 'T2'  # t2-tse sequence
    FLAIR = 3, 'FLAIR'  # FLAIR or t2-tirm sequences
    DWI = 4, 'DWI'  # DWI

    def __str__(self):
        return self.string


@unique
class CTSequenceType(Enum):
    """
    @TODO. What are the actual sequence types for a CT?
    """
    _init_ = 'value string'

    HR = 0, 'High-resolution'  # High-resolution, implying contrast-enhanced

    def __str__(self):
        return self.string


class RadiologicalVolume:
    """
    Class defining how a radiological volume should be handled.
    """
    _unique_id = None  # Internal unique identifier for the radiological volume
    _raw_input_filepath = None  # Original volume filepath on the user's machine
    _usable_input_filepath = None  #
    _output_folder = None  #
    _radiological_type = None  # Disambiguation between CT/MRI, to select from RadiologicalType
    _sequence_type = None  # Specific sequence type within the radiological type
    _timestamp_id = None  # Internal identifier for the corresponding timestamp
    _registered_volumes = {}  # Each element is a dict with the 'filepath' of the registered volume and
    # the 'registration_uid' of the registration applied. The keys are the destination space uid ('MNI' if atlas).
    # @TODO. Do we have a similar dict for the registered atlas files?

    def __init__(self, uid: str, input_filename: str, timestamp_uid: str) -> None:
        self.__reset()
        self._unique_id = uid
        self._raw_input_filepath = input_filename
        self._timestamp_id = timestamp_uid
        self.__init_from_scratch()

    def __reset(self):
        """
        All objects share class or static variables.
        An instance or non-static variables are different for different objects (every object has a copy).
        """
        self._unique_id = None
        self._raw_input_filepath = None
        self._usable_input_filepath = None
        self._output_folder = None
        self._radiological_type = None
        self._sequence_type = None
        self._timestamp_id = None
        self._registered_volumes = {}

    def get_unique_id(self) -> str:
        return self._unique_id

    def get_output_folder(self) -> str:
        return self._output_folder

    def get_raw_input_filepath(self) -> str:
        return self._raw_input_filepath

    def get_usable_input_filepath(self) -> str:
        return self._usable_input_filepath

    def get_sequence_type_enum(self) -> Enum:
        return self._sequence_type

    def get_sequence_type_str(self) -> str:
        return str(self._sequence_type)

    def set_sequence_type(self, type: str) -> None:
        """
        Update the radiological volume sequence type.

        Parameters
        ----------
        type: str
            New sequence type to associate with the current volume, either a str or SequenceType.
        """
        radiological_type = MRISequenceType
        if self._radiological_type == RadiologicalType.CT:
            radiological_type = CTSequenceType

        if isinstance(type, str):
            ctype = get_type_from_string(radiological_type, type)
            if ctype != -1:
                self._sequence_type = ctype
        elif isinstance(type, radiological_type):
            self._sequence_type = type

    def include_registered_volume(self, filepath: str, registration_uid: str, destination_space_uid: str) -> None:
        self._registered_volumes[destination_space_uid] = {"filepath": filepath, "registration_uid": registration_uid}

    def get_registered_volume_info(self, destination_space_uid: str):
        return self._registered_volumes[destination_space_uid]

    def get_registered_volume_destination_uids(self) -> List[str]:
        return list(self._registered_volumes.keys())

    def __init_from_scratch(self):
        self._output_folder = os.path.join(ResourcesConfiguration.getInstance().output_folder, self._timestamp_id)
        os.makedirs(self._output_folder, exist_ok=True)
        self._usable_input_filepath = input_file_type_conversion(input_filename=self._raw_input_filepath,
                                                                 output_folder=self._output_folder)
        self.__parse_sequence_type()

    def __parse_sequence_type(self):
        base_name = self._unique_id.lower()
        if ResourcesConfiguration.getInstance().diagnosis_task == 'neuro_diagnosis':
            self._radiological_type = RadiologicalType.MRI
            if "t2" in base_name and "tirm" in base_name:
                self._sequence_type = MRISequenceType.FLAIR
            elif "flair" in base_name:
                self._sequence_type = MRISequenceType.FLAIR
            elif "t2" in base_name:
                self._sequence_type = MRISequenceType.T2
            elif "gd" in base_name:
                self._sequence_type = MRISequenceType.T1c
            elif "dwi" in base_name:
                self._sequence_type = MRISequenceType.DWI
            else:
                self._sequence_type = MRISequenceType.T1w
        else:
            self._radiological_type = RadiologicalType.CT
            self._sequence_type = CTSequenceType.HR
