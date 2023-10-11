import os
import nibabel as nib
import pandas as pd
import numpy as np
from nibabel import four_to_three
from .configuration_parser import ResourcesConfiguration


def load_nifti_volume(volume_path):
    nib_volume = nib.load(volume_path)
    if len(nib_volume.shape) > 3:
        if len(nib_volume.shape) == 4: #Common problem
            nib_volume = four_to_three(nib_volume)[0]
        else: #DWI volumes
            nib_volume = nib.Nifti1Image(nib_volume.get_fdata()[:, :, :, 0, 0], affine=nib_volume.affine)

    return nib_volume


def dump_predictions(predictions, parameters, nib_volume, storage_prefix):
    print("Writing predictions to files...")
    naming_suffix = 'pred' if parameters.predictions_reconstruction_method == 'probabilities' else 'labels'
    class_names = parameters.training_class_names

    if len(predictions.shape) == 4:
        for c in range(1, predictions.shape[-1]):
            img = nib.Nifti1Image(predictions[..., c], affine=nib_volume.affine)
            #predictions_output_path = os.path.join(storage_prefix + '-' + naming_suffix + '_class' + str(c) + '.nii.gz')
            predictions_output_path = os.path.join(storage_prefix + '-' + naming_suffix + '_' + class_names[c] + '.nii.gz')
            os.makedirs(os.path.dirname(predictions_output_path), exist_ok=True)
            nib.save(img, predictions_output_path)
    else:
        img = nib.Nifti1Image(predictions, affine=nib_volume.affine)
        predictions_output_path = os.path.join(storage_prefix + '-' + naming_suffix + '_' + 'argmax' + '.nii.gz')
        os.makedirs(os.path.dirname(predictions_output_path), exist_ok=True)
        nib.save(img, predictions_output_path)


def generate_cortical_structures_labels_for_slicer(atlas_name):
    struct_description_df = pd.read_csv(ResourcesConfiguration.getInstance().cortical_structures['MNI'][atlas_name]['Description'])
    struct_description_df_sorted = struct_description_df.sort_values(by=['Label'], ascending=True)
    new_values = []
    for index, row in struct_description_df_sorted.iterrows():
        label = row['Label']
        if label == label:
            if atlas_name == 'MNI':
                structure_name = '-'.join(str(row['Region']).strip().split(' '))
                if row['Laterality'] != 'None':
                    structure_name = structure_name + '_' + str(row['Laterality']).strip()
                if row['Matter type'] == 'wm' or row['Matter type'] == 'gm':
                    structure_name = structure_name + '_' + str(row['Matter type']).strip()
            elif atlas_name == 'Harvard-Oxford':
                structure_name = '-'.join(row['Region'].strip().split(' '))
            elif atlas_name == 'Schaefer7' or atlas_name == 'Schaefer17':
                structure_name = '-'.join(row['Region'].strip().split(' '))
            else:
                structure_name = '_'.join(row['Region'].strip().split(' '))

            new_values.append([label, structure_name])

    new_values_df = pd.DataFrame(new_values, columns=['label', 'text'])
    return new_values_df


def generate_subcortical_structures_labels_for_slicer(atlas_name):
    """
    Might be a difference in how subcortical structures are handled compared to cortical structures.
    Even if now, the same method content lies here.
    """
    struct_description_df = pd.read_csv(ResourcesConfiguration.getInstance().subcortical_structures['MNI'][atlas_name]['Description'])
    struct_description_df_sorted = struct_description_df.sort_values(by=['Label'], ascending=True)
    new_values = []
    for index, row in struct_description_df_sorted.iterrows():
        label = row['Label']
        if label == label:
            if atlas_name == 'BCB':
                structure_name = row['Region'].strip()
            else:
                structure_name = row['Region'].strip()
            new_values.append([label, structure_name])

    new_values_df = pd.DataFrame(new_values, columns=['label', 'text'])
    return new_values_df


def neuro_cleanup(patient_parameters):
    # @TODO. Should dump it differently/arrange filenames for re-use in either Raidionics or 3DSlicer.
    # Could maybe have the same structure as for Raidionics, and only adjust for a Slicer use.
    pass
    # if self.from_slicer:
    #     shutil.move(os.path.join(self.output_path, 'input_tumor_mask.nii.gz'),
    #                 os.path.join(self.output_path, 'Tumor.nii.gz'))
    #     shutil.move(brain_mask_filepath,
    #                 os.path.join(self.output_path, 'Brain.nii.gz'))
    #     for s in ResourcesConfiguration.getInstance().neuro_features_cortical_structures:
    #         shutil.move(os.path.join(self.output_path, 'Cortical-structures/' + s + '_mask_to_input.nii.gz'),
    #                     os.path.join(self.output_path, s + '.nii.gz'))
    #         # shutil.move(os.path.join(self.output_path, 'Cortical-structures/MNI_mask_to_input.nii.gz'),
    #         #             os.path.join(self.output_path, 'MNI.nii.gz'))
    #         # shutil.move(os.path.join(self.output_path, 'Cortical-structures/Schaefer7_mask_to_input.nii.gz'),
    #         #             os.path.join(self.output_path, 'Schaefer7.nii.gz'))
    #         # shutil.move(os.path.join(self.output_path, 'Cortical-structures/Schaefer17_mask_to_input.nii.gz'),
    #         #             os.path.join(self.output_path, 'Schaefer17.nii.gz'))
    #         # shutil.move(os.path.join(self.output_path, 'Cortical-structures/Harvard-Oxford_mask_to_input.nii.gz'),
    #         #             os.path.join(self.output_path, 'Harvard-Oxford.nii.gz'))
    #     for s in ResourcesConfiguration.getInstance().neuro_features_subcortical_structures:
    #         shutil.move(os.path.join(self.output_path, 'Subcortical-structures/' + s + '_mask_to_input.nii.gz'),
    #                     os.path.join(self.output_path, s + '.nii.gz'))
    #         # shutil.move(os.path.join(self.output_path, 'Subcortical-structures/BCB_mask_to_input.nii.gz'),
    #         #             os.path.join(self.output_path, 'BCB.nii.gz'))
    #     shutil.move(os.path.join(self.output_path, 'neuro_diagnosis_report.txt'),
    #                 os.path.join(self.output_path, 'Diagnosis.txt'))
    #     shutil.move(os.path.join(self.output_path, 'neuro_diagnosis_report.json'),
    #                 os.path.join(self.output_path, 'Diagnosis.json'))
    #     shutil.move(os.path.join(self.output_path, 'neuro_diagnosis_report.csv'),
    #                 os.path.join(self.output_path, 'Diagnosis.csv'))
    #     os.remove(os.path.join(self.output_path, 'input_to_mni.nii.gz'))
    #     os.remove(os.path.join(self.output_path, 'input_tumor_to_mni.nii.gz'))
    #     self.__generate_cortical_structures_description_file_slicer()
    #     self.__generate_subcortical_structures_description_file_slicer()
    # else:
    #     atlas_desc_dir = os.path.join(self.output_path, 'atlas_descriptions')
    #     os.makedirs(atlas_desc_dir, exist_ok=True)
    #     atlases = ['MNI', 'Schaefer7', 'Schaefer17', 'Harvard-Oxford']
    #     for a in atlases:
    #         df = generate_cortical_structures_labels_for_slicer(atlas_name=a)
    #         output_filename = os.path.join(atlas_desc_dir, a + '_description.csv')
    #         df.to_csv(output_filename)
    #     atlases = ['BCB']  # 'BrainLab'
    #     for a in atlases:
    #         df = generate_subcortical_structures_labels_for_slicer(atlas_name=a)
    #         output_filename = os.path.join(atlas_desc_dir, a + '_description.csv')
    #         df.to_csv(output_filename)
    #     shutil.move(src=os.path.join(self.output_path, 'input_tumor_mask.nii.gz'),
    #                 dst=os.path.join(self.output_path, 'patient', 'input_tumor_mask.nii.gz'))
    #     shutil.move(src=brain_mask_filepath,
    #                 dst=os.path.join(self.output_path, 'patient', 'input_brain_mask.nii.gz'))