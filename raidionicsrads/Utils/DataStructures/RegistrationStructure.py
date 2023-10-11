import os
from aenum import Enum, unique
import shutil
from typing import List
from ..utilities import get_type_from_string, input_file_type_conversion
from ..configuration_parser import ResourcesConfiguration


class Registration:
    """
    Class defining how a registration should be handled.
    """
    _unique_id = None  # Internal unique identifier for the registration instance
    _forward_filepaths = []
    _inverse_filepaths = []
    _output_folder = None  #
    _fixed_uid = None
    _moving_uid = None

    def __init__(self, uid: str, fixed_uid: str, moving_uid: str, fwd_paths: List[str], inv_paths: List[str],
                 output_folder: str) -> None:
        self.__reset()
        self._unique_id = uid
        self._fixed_uid = fixed_uid
        self._moving_uid = moving_uid
        self._output_folder = os.path.join(output_folder, 'Transforms', self._moving_uid + "-to-" + self._fixed_uid)
        os.makedirs(self._output_folder)

        for elem in fwd_paths:
            dest_name = os.path.join(self._output_folder, 'forward_' + os.path.basename(elem))
            shutil.copyfile(elem, dest_name)
            self._forward_filepaths.append(dest_name)
        for elem in inv_paths:
            dest_name = os.path.join(self._output_folder, 'inverse_' + os.path.basename(elem))
            shutil.copyfile(elem, dest_name)
            self._inverse_filepaths.append(dest_name)

    def __reset(self):
        """
        All objects share class or static variables.
        An instance or non-static variables are different for different objects (every object has a copy).
        """
        self._unique_id = None
        self._forward_filepaths = []
        self._inverse_filepaths = []
        self._output_folder = None
        self._fixed_uid = None
        self._moving_uid = None

    def get_unique_id(self) -> str:
        return self._unique_id

    def get_fixed_uid(self) -> str:
        return self._fixed_uid

    def get_moving_uid(self) -> str:
        return self._moving_uid

    def get_output_folder(self) -> str:
        return self._output_folder
