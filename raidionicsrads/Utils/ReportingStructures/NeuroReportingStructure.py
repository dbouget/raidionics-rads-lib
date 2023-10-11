import logging
import traceback
import os
import numpy as np
import operator
import json
import pandas as pd
import collections
from ..configuration_parser import ResourcesConfiguration
from ..io import generate_cortical_structures_labels_for_slicer, generate_subcortical_structures_labels_for_slicer


class NeuroReportingStructure:
    """
    Reporting at a single timestamp with characteristics/features for the tumor.
    """
    _unique_id = None  # Internal unique identifier for the report
    _radiological_volume_uid = None  # Parent CT/MRI volume to which the report is attached
    _output_folder = None
    _tumor_type = None  # Type of brain tumor identified
    _tumor_multifocal = False  # Boolean status about multifocality
    _tumor_parts = 0  #
    _tumor_multifocal_distance = -1.  #
    _statistics = {}

    def __init__(self, id: str, parent_uid: str, output_folder: str):
        """
        """
        self.__reset()
        self._unique_id = id
        self._radiological_volume_uid = parent_uid
        self._output_folder = output_folder
        self._statistics['Main'] = {}
        self._statistics['Main']['Overall'] = TumorStatistics()
        self._statistics['Main']['CoM'] = TumorStatistics()

    def __reset(self):
        """
        All objects share class or static variables.
        An instance or non-static variables are different for different objects (every object has a copy).
        """
        self._unique_id = None
        self._radiological_volume_uid = None
        self._output_folder = None
        self._tumor_type = None
        self._tumor_multifocal = False
        self._tumor_parts = 0
        self._tumor_multifocal_distance = -1.
        self._statistics = {}

    def setup(self, tumor_type: str, tumor_elements: int) -> None:
        self._tumor_type = tumor_type
        self._tumor_parts = tumor_elements
        if tumor_elements > 1:
            self._tumor_multifocal = True
        else:
            self._tumor_multifocal = False

        if self._tumor_multifocal:
            for p in range(tumor_elements):
                self._statistics[str(p+1)] = {}
                self._statistics[str(p+1)]['Overall'] = TumorStatistics()
                self._statistics[str(p+1)]['CoM'] = TumorStatistics()

    def to_txt(self) -> None:
        """

        Exporting the computed tumor characteristics and standardized report in filename.

        Parameters
        ----------

        Returns
        -------
        None
        """
        try:
            filename = os.path.join(self._output_folder, "neuro_clinical_report.txt")
            logging.info("Exporting neuro-parameters to text in {}.".format(filename))
            pfile = open(filename, 'a')
            pfile.write('########### Raidionics clinical report ###########\n')
            pfile.write('Tumor type: {}\n'.format(self._tumor_type))
            pfile.write('Tumor multifocality: {}\n'.format(self._tumor_multifocal))
            pfile.write('  * Number tumor parts: {}\n'.format(self._tumor_parts))
            pfile.write('  * Largest distance between components: {} (mm)\n'.format(np.round(self._tumor_multifocal_distance, 2)))

            pfile.write('\nVolumes\n')
            if self._statistics['Main']['Overall'].original_space_tumor_volume:
                pfile.write('  * Original space: {} (ml)\n'.format(np.round(self._statistics['Main']['Overall'].original_space_tumor_volume, 2)))
            if self._statistics['Main']['Overall'].mni_space_tumor_volume:
                pfile.write('  * MNI space: {} (ml)\n'.format(self._statistics['Main']['Overall'].mni_space_tumor_volume))

            pfile.write('\nLaterality\n')
            pfile.write('  * Left hemisphere: {}%\n'.format(self._statistics['Main']['Overall'].left_laterality_percentage))
            pfile.write('  * Right hemisphere: {}%\n'.format(self._statistics['Main']['Overall'].right_laterality_percentage))
            pfile.write('  * Midline crossing: {}\n'.format(self._statistics['Main']['Overall'].laterality_midline_crossing))

            if self._tumor_type == 'Glioblastoma':
                pfile.write('\nResectability\n')
                pfile.write('  * Expected residual volume: {} (ml)\n'.format(np.round(self._statistics['Main']['Overall'].mni_space_expected_residual_tumor_volume, 2)))
                pfile.write('  * Resection index: {}\n'.format(np.round(self._statistics['Main']['Overall'].mni_space_resectability_index, 3)))

            pfile.write('\nCortical structures overlap\n')
            for t in self._statistics['Main']['Overall'].mni_space_cortical_structures_overlap.keys():
                pfile.write('  * {} atlas\n'.format(t))
                lobes_ordered = collections.OrderedDict(sorted(self._statistics['Main']['Overall'].mni_space_cortical_structures_overlap[t].items(), key=operator.itemgetter(1), reverse=True))
                for r in lobes_ordered.keys():
                    if lobes_ordered[r] != 0:
                        lobe_name = ' '.join(r.lower().replace('main', '').split('_')[:])
                        pfile.write('    - {}: {}%\n'.format(lobe_name, lobes_ordered[r]))

            pfile.write('\nSubcortical structures overlap or distance\n')
            for t in self._statistics['Main']['Overall'].mni_space_subcortical_structures_overlap.keys():
                pfile.write('  * {} atlas\n'.format(t))
                tracts_ordered = collections.OrderedDict(sorted(self._statistics['Main']['Overall'].mni_space_subcortical_structures_overlap[t].items(), key=operator.itemgetter(1), reverse=True))
                for r in tracts_ordered.keys():
                    if tracts_ordered[r] != 0:
                        tract_name = ' '.join(r.lower().replace('main', '').replace('mni', '').split('.')[0].split('_'))
                        pfile.write('    - {}: {}% overlap\n'.format(tract_name, np.round(tracts_ordered[r], 2)))

                pfile.write('\n')
                tracts_ordered = collections.OrderedDict(sorted(self._statistics['Main']['Overall'].mni_space_subcortical_structures_distance[t].items(), key=operator.itemgetter(1), reverse=False))
                for r in tracts_ordered.keys():
                    if tracts_ordered[r] != -1.:
                        tract_name = ' '.join(r.lower().replace('main', '').replace('mni', '').split('.')[0].split('_'))
                        pfile.write('    - {}: {}mm away\n'.format(tract_name, np.round(tracts_ordered[r], 2)))

            # Parameters for each tumor element
            # if self.tumor_multifocal:
            #     for p in range(self.tumor_parts):
            #         tumor_component = str(p+1)
            #         pfile.write('\nTumor part {}\n'.format(tumor_component))

            pfile.close()
        except Exception as e:
            logging.error("Neuro-parameters export to text failed with {}".format(traceback.format_exc()))
        return

    def to_json(self) -> None:
        # @TODO. to modify
        try:
            filename = os.path.join(self._output_folder, "neuro_clinical_report.json")
            logging.info("Exporting neuro-parameters to json in {}.".format(filename))
            param_json = {}
            param_json['Overall'] = {}
            # param_json['Overall']['Presence'] = self._tumor_presence_state
            # if not self._tumor_presence_state:
            #     with open(filename, 'w') as outfile:
            #         json.dump(param_json, outfile)
            #     return

            param_json['Overall']['Type'] = self._tumor_type
            param_json['Overall']['Multifocality'] = self._tumor_multifocal
            param_json['Overall']['Tumor parts nb'] = self._tumor_parts
            param_json['Overall']['Multifocal distance (mm)'] = np.round(self._tumor_multifocal_distance, 2)

            param_json['Main'] = {}
            # param_json['Main']['CenterOfMass'] = {}
            # param_json['Main']['CenterOfMass']['Laterality'] = self.statistics['Main']['CoM'].laterality
            # param_json['Main']['CenterOfMass']['Laterality_perc'] = self.statistics['Main']['CoM'].laterality_percentage
            # param_json['Main']['CenterOfMass']['Lobe'] = []
            # for l in self.statistics['Main']['CoM'].mni_space_cortical_structures_overlap.keys():
            #     param_json['Main']['CenterOfMass']['Lobe'].append([l, self.statistics['Main']['CoM'].mni_space_cortical_structures_overlap[l]])

            param_json['Main']['Total'] = {}
            param_json['Main']['Total']['Volume original (ml)'] = self._statistics['Main']['Overall'].original_space_tumor_volume
            param_json['Main']['Total']['Volume in MNI (ml)'] = self._statistics['Main']['Overall'].mni_space_tumor_volume
            param_json['Main']['Total']['Left laterality (%)'] = self._statistics['Main']['Overall'].left_laterality_percentage
            param_json['Main']['Total']['Right laterality (%)'] = self._statistics['Main']['Overall'].right_laterality_percentage
            param_json['Main']['Total']['Midline crossing'] = self._statistics['Main']['Overall'].laterality_midline_crossing

            if self._tumor_type == 'Glioblastoma':
                param_json['Main']['Total']['ExpectedResidualVolume (ml)'] = np.round(self._statistics['Main']['Overall'].mni_space_expected_residual_tumor_volume, 2)
                param_json['Main']['Total']['ResectionIndex'] = np.round(self._statistics['Main']['Overall'].mni_space_resectability_index, 3)

            param_json['Main']['Total']['CorticalStructures'] = {}
            for t in self._statistics['Main']['Overall'].mni_space_cortical_structures_overlap.keys():
                param_json['Main']['Total']['CorticalStructures'][t] = {}
                for r in self._statistics['Main']['Overall'].mni_space_cortical_structures_overlap[t].keys():
                    param_json['Main']['Total']['CorticalStructures'][t][r] = self._statistics['Main']['Overall'].mni_space_cortical_structures_overlap[t][r]

            param_json['Main']['Total']['SubcorticalStructures'] = {}

            for t in self._statistics['Main']['Overall'].mni_space_subcortical_structures_overlap.keys():
                param_json['Main']['Total']['SubcorticalStructures'][t] = {}
                param_json['Main']['Total']['SubcorticalStructures'][t]['Overlap'] = {}
                param_json['Main']['Total']['SubcorticalStructures'][t]['Distance'] = {}
                for ov in self._statistics['Main']['Overall'].mni_space_subcortical_structures_overlap[t].keys():
                    param_json['Main']['Total']['SubcorticalStructures'][t]['Overlap'][ov] = self._statistics['Main']['Overall'].mni_space_subcortical_structures_overlap[t][ov]
                for di in self._statistics['Main']['Overall'].mni_space_subcortical_structures_distance[t].keys():
                    param_json['Main']['Total']['SubcorticalStructures'][t]['Distance'][di] = self._statistics['Main']['Overall'].mni_space_subcortical_structures_distance[t][di]

            # Parameters for each tumor element
            # if self.tumor_multifocal:
            #     for p in range(self.tumor_parts):
            #         tumor_component = str(p+1)
            #         param_json[tumor_component] = {}
            #         param_json[tumor_component]['CenterOfMass'] = {}
            #         param_json[tumor_component]['CenterOfMass']['Laterality'] = self.statistics[tumor_component]['CoM'].laterality
            #         param_json[tumor_component]['CenterOfMass']['Laterality_perc'] = self.statistics[tumor_component]['CoM'].laterality_percentage
            #         param_json[tumor_component]['CenterOfMass']['Lobe'] = []
            #         for l in self.statistics[tumor_component]['CoM'].mni_space_cortical_structures_overlap.keys():
            #             param_json[tumor_component]['CenterOfMass']['Lobe'].append([l, self.statistics[tumor_component]['CoM'].mni_space_cortical_structures_overlap[l]])
            #
            #         param_json[tumor_component]['Total'] = {}
            #         param_json[tumor_component]['Total']['Volume'] = self.statistics[tumor_component]['Overall'].mni_space_tumor_volume
            #         param_json[tumor_component]['Total']['Laterality'] = self.statistics[tumor_component]['Overall'].laterality
            #         param_json[tumor_component]['Total']['Laterality_perc'] = np.round(self.statistics[tumor_component]['Overall'].laterality_percentage * 100., 2)
            #         param_json[tumor_component]['Total']['Resectability'] = np.round(self.statistics[tumor_component]['Overall'].mni_space_resectability_score * 100., 2)
            #
            #         param_json[tumor_component]['Total']['Lobe'] = []
            #
            #         for l in self.statistics[tumor_component]['Overall'].mni_space_cortical_structures_overlap.keys():
            #             param_json[tumor_component]['Total']['Lobe'].append([l, self.statistics[tumor_component]['Overall'].mni_space_cortical_structures_overlap[l]])
            #
            #         param_json[tumor_component]['Total']['Tract'] = {}
            #         param_json[tumor_component]['Total']['Tract']['Distance'] = []
            #
            #         for l in self.statistics[tumor_component]['Overall'].mni_space_tracts_distance.keys():
            #             if self.statistics[tumor_component]['Overall'].mni_space_tracts_distance[l] != -1.:
            #                 param_json[tumor_component]['Total']['Tract']['Distance'].append([l, np.round(self.statistics[tumor_component]['Overall'].mni_space_tracts_distance[l], 2)])
            #
            #         param_json[tumor_component]['Total']['Tract']['Overlap'] = []
            #         for l in self.statistics[tumor_component]['Overall'].mni_space_tracts_overlap.keys():
            #             if self.statistics[tumor_component]['Overall'].mni_space_tracts_overlap[l] > 0:
            #                 param_json[tumor_component]['Total']['Tract']['Overlap'].append([l, np.round(self.statistics[tumor_component]['Overall'].mni_space_tracts_overlap[l] * 100., 2)])

            with open(filename, 'w', newline='\n') as outfile:
                json.dump(param_json, outfile, indent=4, sort_keys=True)
        except Exception as e:
            logging.error("Neuro-parameters export to json failed with {}".format(traceback.format_exc()))

        return

    def to_csv(self) -> None:
        try:
            filename = os.path.join(self._output_folder, "neuro_clinical_report.csv")
            logging.info("Exporting neuro-parameters to csv in {}.".format(filename))

            # if not self._tumor_presence_state:
            #     return
            values = [self._tumor_multifocal, self._tumor_parts, np.round(self._tumor_multifocal_distance, 2)]
            column_names = ['Multifocality', 'Tumor parts nb', 'Multifocal distance (mm)']

            values.extend([self._statistics['Main']['Overall'].original_space_tumor_volume, self._statistics['Main']['Overall'].mni_space_tumor_volume])
            column_names.extend(['Volume original (ml)', 'Volume in MNI (ml)'])

            values.extend([self._statistics['Main']['Overall'].left_laterality_percentage,
                           self._statistics['Main']['Overall'].right_laterality_percentage,
                           self._statistics['Main']['Overall'].laterality_midline_crossing])
            column_names.extend(['Left laterality (%)', 'Right laterality (%)', 'Midline crossing'])

            if self._tumor_type == 'Glioblastoma':
                values.extend([np.round(self._statistics['Main']['Overall'].mni_space_expected_residual_tumor_volume, 2),
                               np.round(self._statistics['Main']['Overall'].mni_space_resectability_index, 3)])
                column_names.extend(['ExpectedResidualVolume (ml)', 'ResectionIndex'])

            for t in self._statistics['Main']['Overall'].mni_space_cortical_structures_overlap.keys():
                for r in self._statistics['Main']['Overall'].mni_space_cortical_structures_overlap[t].keys():
                    values.extend([self._statistics['Main']['Overall'].mni_space_cortical_structures_overlap[t][r]])
                    column_names.extend([r.split('.')[0].lower().strip() + '_overlap'])

            for t in self._statistics['Main']['Overall'].mni_space_subcortical_structures_overlap.keys():
                for r in self._statistics['Main']['Overall'].mni_space_subcortical_structures_overlap[t].keys():
                    values.extend([self._statistics['Main']['Overall'].mni_space_subcortical_structures_overlap[t][r]])
                    column_names.extend([t + r.split('.')[0][:-4] + '_overlap'])

            for t in self._statistics['Main']['Overall'].mni_space_subcortical_structures_distance.keys():
                for r in self._statistics['Main']['Overall'].mni_space_subcortical_structures_overlap[t].keys():
                    values.extend([self._statistics['Main']['Overall'].mni_space_subcortical_structures_distance[t][r]])
                    column_names.extend([t + r.split('.')[0][:-4] + '_distance'])

            values_df = pd.DataFrame(np.asarray(values).reshape((1, len(values))), columns=column_names)
            values_df.to_csv(filename, index=False)
        except Exception as e:
            logging.error("Neuro-parameters export to csv failed with {}".format(traceback.format_exc()))

    def dump_descriptions(self):
        atlas_desc_dir = os.path.join(self._output_folder, 'atlas_descriptions')
        os.makedirs(atlas_desc_dir, exist_ok=True)
        atlases = ResourcesConfiguration.getInstance().neuro_features_cortical_structures #['MNI', 'Schaefer7', 'Schaefer17', 'Harvard-Oxford']
        for a in atlases:
            df = generate_cortical_structures_labels_for_slicer(atlas_name=a)
            output_filename = os.path.join(atlas_desc_dir, a + '_description.csv')
            df.to_csv(output_filename)
        atlases = ResourcesConfiguration.getInstance().neuro_features_subcortical_structures #['BCB']  # 'BrainLab'
        for a in atlases:
            df = generate_subcortical_structures_labels_for_slicer(atlas_name=a)
            output_filename = os.path.join(atlas_desc_dir, a + '_description.csv')
            df.to_csv(output_filename)


class TumorStatistics:
    """
    Specific class for holding the computed tumor characteristics/features.
    """
    def __init__(self):
        self.left_laterality_percentage = None
        self.right_laterality_percentage = None
        self.laterality_midline_crossing = None
        self.original_space_tumor_volume = None
        self.mni_space_tumor_volume = None
        self.mni_space_expected_resectable_tumor_volume = None
        self.mni_space_expected_residual_tumor_volume = None
        self.mni_space_resectability_index = None
        self.mni_space_cortical_structures_overlap = {}
        self.mni_space_subcortical_structures_overlap = {}
        self.mni_space_subcortical_structures_distance = {}
