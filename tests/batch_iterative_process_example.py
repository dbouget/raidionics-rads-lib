import os
import json
import shutil
import configparser
import logging
import sys
import subprocess
import traceback
import argparse
import platform


def batch_iterative_process_example():
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
    parser.add_argument('--input', metavar='input', help='Path to the input patients folder')
    parser.add_argument('--output', metavar='output', help='Path to save the predictions')
    parser.add_argument('--models', metavar='models', help='Path to the trained models folder')
    parser.add_argument('--backend', metavar='backend', help='Favored processing approach, either venv or Docker',
                        choices=['local', 'docker'])
    parser.add_argument('--verbose', help="To specify the level of verbose, Default: warning", type=str,
                        choices=['debug', 'info', 'warning', 'error'], default='warning')

    argsin = sys.argv[1:]
    args = parser.parse_args(argsin)
    cohort_folderpath = args.input
    dest_folderpath = args.output
    models_folderpath = args.models
    process_backend = args.backend

    logging.basicConfig()
    logging.getLogger().setLevel(logging.WARNING)

    if args.verbose == 'debug':
        logging.getLogger().setLevel(logging.DEBUG)
    elif args.verbose == 'info':
        logging.getLogger().setLevel(logging.INFO)
    elif args.verbose == 'error':
        logging.getLogger().setLevel(logging.ERROR)

    patients = []
    for _, dirs, _ in os.walk(cohort_folderpath):
        for d in dirs:
            patients.append(d)
        break

    for pat in patients:
        tmp_folder = ''
        try:
            # Setting up directories
            input_pat_folder = os.path.join(cohort_folderpath, pat)
            dest_pat_folder = os.path.join(dest_folderpath, pat)
            if os.path.exists(dest_pat_folder):
                print("Skipping inference for patient {}. Delete destination folder beforehand".format(pat))
                continue

            os.makedirs(dest_pat_folder)
            tmp_folder = os.path.join(dest_folderpath, 'tmp')
            if os.path.exists(tmp_folder):
                shutil.rmtree(tmp_folder)
            os.makedirs(tmp_folder)

            # Setting up the configuration file
            rads_config = configparser.ConfigParser()
            rads_config.add_section('Default')
            rads_config.set('Default', 'task', 'neuro_diagnosis')
            rads_config.set('Default', 'caller', '')
            rads_config.add_section('System')
            rads_config.set('System', 'gpu_id', "-1")
            rads_config.set('System', 'input_folder', input_pat_folder)
            rads_config.set('System', 'output_folder', dest_pat_folder)
            rads_config.set('System', 'model_folder', models_folderpath)
            rads_config.set('System', 'pipeline_filename', os.path.join(models_folderpath, 'MRI_Tumor_Postop', 'pipeline.json'))
            rads_config.add_section('Runtime')
            rads_config.set('Runtime', 'reconstruction_method', 'probabilities')  # thresholding, probabilities
            rads_config.set('Runtime', 'reconstruction_order', 'resample_first')
            rads_config.set('Runtime', 'use_preprocessed_data', 'False')

            # Running the process
            if process_backend == 'local':
                rads_config_filename = os.path.join(tmp_folder, 'rads_config.ini')
                with open(rads_config_filename, 'w') as outfile:
                    rads_config.write(outfile)
                if platform.system() == 'Windows':
                    subprocess.check_call(['raidionicsrads',
                                           '{config}'.format(config=rads_config_filename),
                                           '--verbose', args.verbose], shell=True)
                else:
                    subprocess.check_call(['raidionicsrads',
                                           '{config}'.format(config=rads_config_filename),
                                           '--verbose', args.verbose])
            elif process_backend == 'docker':
                # @TODO. For using Docker, models and pipelines should also be copied in a temp input folder to mount.
                # In addition, the paths going into the config should be based on the /home/ubuntu/resources mounted space.
                docker_folder = os.path.join(tmp_folder, 'docker')
                os.makedirs(docker_folder)

                shutil.copytree(src=models_folderpath, dst=os.path.join(docker_folder, 'models'))
                shutil.copytree(src=input_pat_folder, dst=os.path.join(docker_folder, 'inputs'))
                os.makedirs(os.path.join(docker_folder, 'outputs'))

                rads_config.set('System', 'input_folder', '/home/ubuntu/resources/inputs')
                rads_config.set('System', 'output_folder', '/home/ubuntu/resources/outputs')
                rads_config.set('System', 'model_folder', '/home/ubuntu/resources/models')
                rads_config.set('System', 'pipeline_filename',
                                '/home/ubuntu/resources/models/MRI_Tumor_Postop/pipeline.json')
                rads_config_filename = os.path.join(docker_folder, 'rads_config.ini')
                with open(rads_config_filename, 'w') as outfile:
                    rads_config.write(outfile)
                cmd_docker = ['docker', 'run', '-v', '{}:/home/ubuntu/resources'.format(docker_folder),
                              '--runtime=nvidia', '--network=host', '--ipc=host', 'dbouget/raidionics-rads:v1.1',
                              '-c /home/ubuntu/resources/rads_config.ini', '-v', args.verbose]
                if platform.system() == 'Windows':
                    subprocess.check_call(cmd_docker, shell=True)
                else:
                    subprocess.check_call(cmd_docker)
            else:
                logging.error("Backend option not supported, please select from [local, docker]")
                return

            # Clean-up
            if os.path.exists(tmp_folder):
                shutil.rmtree(tmp_folder)
        except Exception:
            print("Patient {} failed.".format(pat))
            print(traceback.format_exc())
            if os.path.exists(tmp_folder):
                shutil.rmtree(tmp_folder)

            continue


batch_iterative_process_example()
