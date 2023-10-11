import logging

from aenum import Enum, unique
from typing import Union
import os
import SimpleITK as sitk
import numpy as np


def get_type_from_string(enum_type: Enum, string: str) -> Union[str, int]:
    if type(string) == str:
        for i in range(len(list(enum_type))):
            #if string == list(EnumType)[i].name:
            if string == str(list(enum_type)[i]):
                return list(enum_type)[i]
        return -1
    elif type(string) == enum_type:
        return string
    else: #Unmanaged input type
        logging.warning("Unable to find a match for type {} and value {}.".format(enum_type, string))
        return -1


def get_type_from_enum_name(enum_type: Enum, string: str) -> Union[str, int]:
    if type(string) == str:
        for i in range(len(list(enum_type))):
            if string.lower() == list(enum_type)[i].name.lower():
                return list(enum_type)[i]
        return -1
    elif type(string) == enum_type:
        return string
    else: #Unmanaged input type
        logging.warning("Unable to find a match for type {} and value {}.".format(enum_type, string))
        return -1


def input_file_category_disambiguation(input_filename: str) -> str:
    """
    Identifying whether the volume stored on disk under input_filename contains a radiological volume or is an
    integer-like volume with labels.
    The category belongs to [Volume, Annotation].

    Parameters
    ----------
    input_filename: str
        Disk location of the volume to disambiguate.

    Returns
    ----------
    str
        Human-readable category identified for the input.
    """
    category = None
    reader = sitk.ImageFileReader()
    reader.SetFileName(input_filename)
    image = reader.Execute()
    image_type = image.GetPixelIDTypeAsString()
    array = sitk.GetArrayFromImage(image)

    if len(np.unique(array)) > 255 or np.max(array) > 255 or np.min(array) < -1:
        category = "Volume"
    # If the input radiological volume has values within [0, 255] only. Empirical solution for now, since less than
    # 10 classes are usually handle at any given time.
    elif len(np.unique(array)) >= 25:
        category = "Volume"
    else:
        category = "Annotation"
    return category


def input_file_type_conversion(input_filename: str, output_folder: str) -> str:
    # Always converting the input file to compressed nifti (based on SimpleITK), otherwise will be discarded.
    # @TODO. Do we catch a potential .seg file that would be coming from 3D Slicer for annotations?
    pre_file_extension = os.path.basename(input_filename).split('.')[0]
    file_extension = '.'.join(os.path.basename(input_filename).split('.')[1:])
    filename = input_filename
    if file_extension != 'nii.gz':
        input_sitk = sitk.ReadImage(input_filename)
        nifti_outfilename = os.path.join(output_folder, pre_file_extension + '.nii.gz')
        sitk.WriteImage(input_sitk, nifti_outfilename)
        filename = nifti_outfilename

    return filename
