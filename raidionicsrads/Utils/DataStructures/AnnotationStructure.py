import os
from typing import List
from aenum import Enum, unique
from ..utilities import get_type_from_string, get_type_from_enum_name, input_file_type_conversion
from ..configuration_parser import ResourcesConfiguration


@unique
class AnnotationClassType(Enum):
    """
    Generic enumeration type for describing the type of annotation.
    """
    _init_ = 'value string'

    Brain = 0, 'Brain'
    Tumor = 1, 'Tumor'

    Lungs = 100, 'Lungs'
    Airways = 101, 'Airways'
    LymphNodes = 102, 'Lymph nodes (all sizes)'
    VenaCava = 103, 'Vena cava'
    AorticArch = 104, 'Aortic arch'
    AscendingAorta = 105, 'Ascending aorta'
    DescendingAorta = 106, 'Descending aorta'
    Spine = 107, 'Spine'
    Heart = 108, 'Heart'
    PulmonaryVeins = 109, 'Pulmonary veins'
    PulmonaryTrunk = 110, 'Pulmonary trunk'
    BrachiocephalicVeins = 111, 'Brachiocephalic veins'
    SubCarArt = 112, 'Subclavian and carotid arteries'
    Azygos = 113, 'Azygos vein'
    Esophagus = 114, 'Esophagus'

    def __str__(self):
        return self.string


@unique
class BrainTumorType(Enum):
    """
    Specific enumeration type for brain tumor sub-types
    """
    _init_ = 'value string'

    GBM = 0, 'Glioblastoma'
    LGG = 1, 'Lower-grade glioma'
    Meningioma = 2, 'Meningioma'
    Metastasis = 3, 'Metastasis'

    def __str__(self):
        return self.string


class Annotation:
    """
    Class defining how an annotation should be handled.
    """
    _unique_id = None  # Internal unique identifier for the annotation instance
    _raw_input_filepath = None  # Original volume filepath on the user's machine
    _usable_input_filepath = None  #
    _output_folder = None  #
    _radiological_volume_uid = None
    _annotation_type = None
    _annotation_subtype = None
    _registered_volumes = {}
    # @TODO. Should we save also if the annotation is manual or automatic?

    def __init__(self, uid: str, input_filename: str, output_folder: str, radiological_volume_uid: str,
                 annotation_class: str) -> None:
        self.__reset()
        self._unique_id = uid
        self._raw_input_filepath = input_filename
        self._output_folder = output_folder
        self._radiological_volume_uid = radiological_volume_uid
        self._annotation_type = get_type_from_enum_name(AnnotationClassType, annotation_class)
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
        self._radiological_volume_uid = None
        self._annotation_type = None
        self._annotation_subtype = None
        self._registered_volumes = {}

    def get_unique_id(self) -> str:
        return self._unique_id

    def get_output_folder(self) -> str:
        return self._output_folder

    @property
    def raw_input_filepath(self) -> str:
        return self._raw_input_filepath

    # @raw_input_filepath.setter
    # def raw_input_filepath(self, filepath: str) -> None:
    #     self._raw_input_filepath = filepath
    #     # @TODO. To check, might not want to give the option to change input filepath?
    #     self.__init_from_scratch()

    def get_usable_input_filepath(self) -> str:
        return self._usable_input_filepath

    def get_parent_radiological_volume_uid(self) -> str:
        return self._radiological_volume_uid

    def get_annotation_type_enum(self) -> Enum:
        return self._annotation_type

    def get_annotation_type_str(self) -> str:
        return str(self._annotation_type)

    def set_annotation_type(self, type: str) -> None:
        """
        Update the annotation class type by providing its string version, which will be matched to the proper
        element inside AnnotationClassType.

        Parameters
        ----------
        type: str
            New annotation type to associate with the current instance.
        """
        if isinstance(type, str):
            ctype = get_type_from_string(AnnotationClassType, type)
            if ctype != -1:
                self._annotation_type = ctype
        elif isinstance(type, AnnotationClassType):
            self._annotation_type = type

    def get_annotation_subtype_enum(self) -> Enum:
        return self._annotation_subtype

    def get_annotation_subtype_str(self) -> str:
        return str(self._annotation_subtype)

    def set_annotation_subtype(self, type: Enum, value: str) -> None:
        """


        Parameters
        ----------
        type: Enum
            Specific Enum class where the subtype belongs.
        value: str
            String version of the requested subtype.
        """
        ctype = get_type_from_string(type, value)
        if ctype != -1:
            self._annotation_subtype = ctype

    def include_registered_volume(self, filepath: str, registration_uid: str, destination_space_uid: str) -> None:
        if destination_space_uid in list(self._registered_volumes.keys()):
            raise ValueError("[AnnotationStructure] Trying to insert a registered volume with an already existing "
                             "destination space key: {}.".format(destination_space_uid))
        self._registered_volumes[destination_space_uid] = {"filepath": filepath, "registration_uid": registration_uid}

    def get_registered_volume_info(self, destination_space_uid: str):
        return self._registered_volumes[destination_space_uid]

    def get_registered_volume_destination_uids(self) -> List[str]:
        return list(self._registered_volumes.keys())

    def __init_from_scratch(self):
        """
        Mostly in case the annotation was provided by the user in a non-nifti format.
        """
        self._usable_input_filepath = input_file_type_conversion(input_filename=self._raw_input_filepath,
                                                                 output_folder=self._output_folder)
