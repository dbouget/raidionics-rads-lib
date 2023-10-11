import json
import traceback
import logging
import numpy as np
from .AbstractPipelineStep import AbstractPipelineStep
from ..Utils.configuration_parser import ResourcesConfiguration
from ..Utils.ReportingStructures.NeuroSurgicalReportingStructure import NeuroSurgicalReportingStructure
from ..Processing.neuro_report_computing import compute_surgical_report
from ..Utils.DataStructures.AnnotationStructure import AnnotationClassType


class SurgicalReportingStep(AbstractPipelineStep):
    """

    """
    _patient_parameters = None
    _report = None

    def __init__(self, step_json: dict) -> None:
        super(SurgicalReportingStep, self).__init__(step_json=step_json)
        self.__reset()

    def __reset(self):
        """
        All objects share class or static variables.
        An instance or non-static variables are different for different objects (every object has a copy).
        """
        self._patient_parameters = None
        self._report = None

    def setup(self, patient_parameters):
        self._patient_parameters = patient_parameters

    def execute(self):
        if ResourcesConfiguration.getInstance().diagnosis_task == 'neuro_diagnosis':
            self.__run_neuro_surgical_reporting()
        else:
            pass
        return self._patient_parameters

    def __run_neuro_surgical_reporting(self):
        """

        """
        try:
            non_available_uid = True
            report_uid = None
            while non_available_uid:
                report_uid = 'SurRep' + str(np.random.randint(0, 10000))
                if report_uid not in self._patient_parameters.get_all_reportings_uids():
                    non_available_uid = False
            report = NeuroSurgicalReportingStructure(id=report_uid,
                                                     output_folder=ResourcesConfiguration.getInstance().output_folder)
            preop_t1ce_uid = self._patient_parameters.get_radiological_volume_uid(timestamp=0, sequence="T1-CE")
            postop_t1ce_uid = self._patient_parameters.get_radiological_volume_uid(timestamp=1, sequence="T1-CE")
            preop_tumor_uid = self._patient_parameters.get_all_annotations_uids_class_radiological_volume(
                volume_uid=preop_t1ce_uid, annotation_class=AnnotationClassType.Tumor)
            postop_tumor_uid = self._patient_parameters.get_all_annotations_uids_class_radiological_volume(
                volume_uid=postop_t1ce_uid, annotation_class=AnnotationClassType.Tumor)
            if len(preop_tumor_uid) > 0 and len(postop_tumor_uid) > 0:
                compute_surgical_report(self._patient_parameters.get_annotation(preop_tumor_uid[0]).get_usable_input_filepath(),
                                        self._patient_parameters.get_annotation(postop_tumor_uid[0]).get_usable_input_filepath(), report)
            self._patient_parameters.include_reporting(report_uid, report)
            report.to_json()
        except Exception as e:
            logging.error("[SurgicalReportingStep] Neuro surgical reporting failed with: {}.".format(traceback.format_exc()))
            raise ValueError("[SurgicalReportingStep] Neuro surgical reporting failed.")
