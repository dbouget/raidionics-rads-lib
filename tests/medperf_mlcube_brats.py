import os

import numpy as np
import yaml
import shutil
import configparser
import logging
import logging.config
import sys
from enum import Enum
import uuid
import subprocess
import traceback
import argparse
import platform
import nibabel as nib
from copy import deepcopy


def mlcube_brats(exec_id, task_args, model_name):
    """
    Example code for running a pipeline iteratively over a set of patients. Custom pipelines can be performed by
    feeding a local json file to the following line below in the code:
    rads_config.set('System', 'pipeline_filename', os.path.join(models_folderpath, 'MRI_Tumor_Postop', 'pipeline.json'))

    All trained models, necessary for running the segmentation tasks selected in the pipeline, must be manually
    downloaded (https://github.com/dbouget/Raidionics-models/releases/tag/1.2.0), extracted and placed within an overall
    models folder.

    Parameters
    ----------
    --input Folder path containing sub-folders, one for each patient, to process. Images inside each patient folder are
    expected to be in nifti format (nii.gz) and their names must contain their MRI sequence type (i.e., t1gd for T1-CE,
    t1 for T1-w, flair for FLAIR) or the -label_target suffix for annotations (i.e., label_tumor) appended to the same
    name as the MRI image it corresponds to.
    --output Destination folder where the processed results will be dumped. Sub-folders, one for each patient, will be
    automatically generated.
    --models Folder path where all trained models are located on disk.
    --backend Indication to either perform all processing directly (assuming located inside a proper venv) or inside
    a Docker container. To chose from [local, docker].
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', metavar='data_dir', help='Path to the input patients folder')
    parser.add_argument('--output_dir', metavar='output_dir', help='Path to save the predictions')

    # argsin = sys.argv[1:]
    # args = parser.parse_args(argsin)
    args = parser.parse_args(task_args)
    input_folderpath = args.data_dir
    dest_folderpath = args.output_dir

    print("Input data folder: {}".format(input_folderpath))
    print("Destination folder: {}".format(dest_folderpath))
    logging.basicConfig()
    logging.getLogger().setLevel(logging.INFO)

    patients = []
    for _, dirs, _ in os.walk(input_folderpath):
        for d in dirs:
            patients.append(d)
        break

    if len(patients) == 0:
        # Not a cohort but a single patient
        patients.append(os.path.basename(input_folderpath))
        input_folderpath = os.path.dirname(input_folderpath)

    for pat in patients:
        print("Processing patient: {}".format(pat))
        tmp_folder = ''
        revamped_input_folder = ''
        dest_pat_folder = ''
        try:
            # Setting up directories
            input_pat_folder = os.path.join(input_folderpath, pat)
            dest_pat_folder = os.path.join(dest_folderpath, pat)
            if os.path.exists(dest_pat_folder):
                print("Skipping inference for patient {}. Delete destination folder beforehand".format(pat))
                continue

            os.makedirs(dest_pat_folder)
            tmp_folder = os.path.join(dest_folderpath, 'tmp')
            if os.path.exists(tmp_folder):
                shutil.rmtree(tmp_folder)
            os.makedirs(tmp_folder)

            # Revamping the input folder for proper naming conventions
            revamped_input_folder = os.path.join(dest_folderpath, 'fix_input')
            logging.info("Running inference on {}".format(revamped_input_folder))
            logging.info("Inference results will be stored in {}".format(dest_pat_folder))

            if os.path.exists(revamped_input_folder):
                shutil.rmtree(revamped_input_folder)
            os.makedirs(revamped_input_folder)
            seq_files = []
            for _, _, files in os.walk(input_pat_folder):
                for f in files:
                    seq_files.append(f)
                break

            print("Identified {} files in patient folder.".format(len(seq_files)))
            os.makedirs(os.path.join(revamped_input_folder, "T0"))
            for sf in seq_files:
                if "t1c" in sf:
                    src_filename = os.path.join(input_pat_folder, sf)
                    new_name = sf.replace("t1c", "t1_gd")
                    shutil.copyfile(src=src_filename,
                                    dst=os.path.join(revamped_input_folder, "T0", new_name))
                if "t1n" in sf:
                    src_filename = os.path.join(input_pat_folder, sf)
                    new_name = sf.replace("t1n", "t1_woc")
                    shutil.copyfile(src=src_filename,
                                    dst=os.path.join(revamped_input_folder, "T0", new_name))
                if "t2f" in sf:
                    src_filename = os.path.join(input_pat_folder, sf)
                    new_name = sf.replace("t2f", "flair")
                    shutil.copyfile(src=src_filename,
                                    dst=os.path.join(revamped_input_folder, "T0", new_name))

                if "t2w" in sf:
                    src_filename = os.path.join(input_pat_folder, sf)
                    new_name = sf.replace("t2w", "t2")
                    shutil.copyfile(src=src_filename,
                                    dst=os.path.join(revamped_input_folder, "T0", new_name))

            # Setting up the configuration file
            rads_config = configparser.ConfigParser()
            rads_config.add_section('Default')
            rads_config.set('Default', 'task', 'neuro_diagnosis')
            rads_config.set('Default', 'caller', '')
            rads_config.add_section('System')
            rads_config.set('System', 'gpu_id', "0")
            rads_config.set('System', 'input_folder', revamped_input_folder)
            rads_config.set('System', 'output_folder', dest_pat_folder)
            rads_config.set('System', 'model_folder', "/workspace/additional_files/models")
            rads_config.set('System', 'pipeline_filename', '/workspace/additional_files/models/' + model_name + '/pipeline.json')
            rads_config.add_section('Runtime')
            rads_config.set('Runtime', 'reconstruction_method', 'thresholding')  # thresholding, probabilities
            rads_config.set('Runtime', 'reconstruction_order', 'resample_first')
            rads_config.set('Runtime', 'use_stripped_data', 'True')
            rads_config.set('Runtime', 'use_registered_data', 'True')

            # Running the process
            rads_config_filename = os.path.join(tmp_folder, 'rads_config.ini')
            with open(rads_config_filename, 'w') as outfile:
                rads_config.write(outfile)
            if platform.system() == 'Windows':
                subprocess.check_call(['raidionicsrads',
                                       '{config}'.format(config=rads_config_filename),
                                       '--verbose', 'info'], shell=True)
            else:
                subprocess.check_call(['raidionicsrads',
                                       '{config}'.format(config=rads_config_filename),
                                       '--verbose', 'info'])
            # Output files reformatting
            prediction_files = []
            for _, _, files in os.walk(os.path.join(dest_pat_folder, "T0")):
                for f in files:
                    prediction_files.append(f)
                break

            global_prediction = None
            global_prediction_affine = None
            for pf in prediction_files:
                predictions_nib = nib.load(os.path.join(dest_pat_folder, "T0", pf))
                predictions = predictions_nib.get_fdata()[:]
                if global_prediction is None:
                    global_prediction = np.zeros(predictions.shape)
                    global_prediction_affine = predictions_nib.affine

                if "Necrosis" in pf:
                    global_prediction[predictions == 1] = 1
                elif "Edema" in pf:
                    global_prediction[predictions == 1] = 2
                elif "Tumor" in pf:
                    global_prediction[predictions == 1] = 3

            global_prediction_filename = os.path.join(dest_folderpath, prediction_files[0].split("-t1")[0] + '.nii.gz')
            nib.save(nib.Nifti1Image(global_prediction, global_prediction_affine), global_prediction_filename)

            # Clean-up
            if os.path.exists(tmp_folder):
                shutil.rmtree(tmp_folder)
            if os.path.exists(revamped_input_folder):
                shutil.rmtree(revamped_input_folder)
            if os.path.exists(dest_pat_folder):
                shutil.rmtree(dest_pat_folder)
        except Exception:
            print("Patient {} failed.".format(pat))
            print(traceback.format_exc())
            if os.path.exists(tmp_folder):
                shutil.rmtree(tmp_folder)
            if os.path.exists(revamped_input_folder):
                shutil.rmtree(revamped_input_folder)
            if os.path.exists(dest_pat_folder):
                shutil.rmtree(dest_pat_folder)
            continue


class Task(str, Enum):
    """Tasks implemented in this MLCube"""

    Gli_Seg = 'gli_seg_brats'
    """Glioma segmentation"""

    Gli_SSA_Seg = 'gli_ssa_seg_brats'
    """Glioma SSA segmentation"""

    Men_Seg = 'men_seg_brats'
    """Meningioma segmentation"""

    Met_Seg = 'met_seg_brats'
    """Metastasis segmentation"""


def main():
    """
    mnist.py task task_specific_parameters...
    """
    # noinspection PyBroadException
    print("Preparsing message")
    parser = argparse.ArgumentParser()
    parser.add_argument('mlcube_task', type=str, help="Task for this MLCube.")
    mlcube_args, task_args = parser.parse_known_args()
    # print("mlcube args: {}".format(mlcube_args))
    # print("task args: {}".format(task_args))

    execution_id = str(uuid.uuid4())
    if mlcube_args.mlcube_task == Task.Gli_Seg:
        mlcube_brats(execution_id, task_args, "MRI_GLI_Brats")
    elif mlcube_args.mlcube_task == Task.Gli_SSA_Seg:
        mlcube_brats(execution_id, task_args, "MRI_GLI_SSA_Brats")
    elif mlcube_args.mlcube_task == Task.Men_Seg:
        mlcube_brats(execution_id, task_args, "MRI_Meningioma_Brats")
    elif mlcube_args.mlcube_task == Task.Met_Seg:
        mlcube_brats(execution_id, task_args, "MRI_Metastasis_Brats")
    else:
        raise ValueError(f"Unknown task: {task_args}")
    print(f"MLCube task ({mlcube_args.mlcube_task}) completed. See log file for details.")


if __name__ == '__main__':
    main()
