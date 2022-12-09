"""
Copyright (c) 2021 Nuance Communications, Inc.

Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
of the Software, and to permit persons to whom the Software is furnished to do
so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

This module exists to isolate logging configuration in a separate name space
This allows these values to be updated completely independent of any other
values in this package

This is an example AI Service.
"""
import glob
import logging
import os
from datetime import datetime
from typing import List

import numpy as np
import pydicom.uid
import torch
from ai_service import AiJobProcessor, AiService, Series, related_uid
from ai_service.utility import AiServiceException
from diagnostic_report import DiagnosticReport
from preprocess import *
from version import VERSION
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader
from pydicom import dcmread
import pydicom
import PIL.ImageOps  
from skimage import exposure
from torchvision.models import resnet50, ResNet50_Weights
import torchvision.transforms as transforms


# -------------------------------------------------------------------------------
class BoneageAiService(AiJobProcessor):

    partner_name = "mgb"
    service_name = "boneage"
    version = VERSION

    def filter_image(self, image: pydicom.Dataset) -> bool:
        """
        The first step of processing a job is to examine DICOM images and
        filter out those images that do not meet necessary criteria. DICOM
        tags such as modality and body part should be considered.
        Please remember that some tags are optional.

        Parameters
        ----------
        image
            A DICOM image as opened by pydicom.dcmread().

        Returns
        -------
            True if this DICOM images should be included in this study.
            False otherwise.
        """
        # Check Modality, ImageType, and ImageOrientation tags
        if image.Modality == "CR" or image.Modality == "DX":
        
            return True
        return False
    

    def select_series(
        self, image_series_list: List[Series]
    ) -> List[Series]:
        """
        After DICOM images have been filtered, they are collated by series.
        AI Services must select series that should be analyzed.
        Many AI services analyze a single series and select the "best" one for analysis.
        Some AI services select multiple series.

        Parameters
        ----------
        image_series_list : List[Series]
            AI Services are passed a list of series.
            Each series is a wrapper around a list of DICOM images as opened by pydicom.dcmread().
            A Series object will behave as a list of DICOM images when iterated over,
            but also has members describing series_uid and study_uid

        Returns
        -------
            List[Series]
            List of Series that meet the appropriate criteria.
            The return value should be the same structure as the input value.
            If no DICOM images meet criteria, return an empty list.

        """
        # Generally, one would filter based upon the most current DICOM file.
        # This example pulls the latest dicom, using the series time and date information,
        # using the study time and date as a backup since series time information is required by DICOM standard
        latest_series = None
        latest_datetime = 0
        for series in image_series_list:
            series_datetime = series.image.get("SeriesDate",
                                               series.image.get("StudyDate"))
            series_datetime += series.image.get("SeriesTime",
                                                series.image.get("StudyTime"))
            series_datetime = int(float(series_datetime))
            if series_datetime > latest_datetime:
                latest_series = series
                latest_datetime = series_datetime
        if latest_series is None:
            return []
        return [latest_series]

    # ---------------------------------------------------------------------------
    @classmethod
    def initialize_class(cls):
        """
        Loads model object.

        This class method will be called after the worker process has been
        forked but before the processor is considered ready. Use it to perform
        initialization that is common for all instances of a class.  Note that
        new processor instances are created for each job.

        This service's model is loaded in this fashion because of the way
        Gunicorn and TensorFlow work.  Tensorflow-Keras requires that models
        are loaded AFTER any process is forked.  However, Gunicorn uses fork
        to create the worker process.

        This method is called AFTER Gunicorn forks the worker process, but
        BEFORE any jobs cause this class to be instantiated.
        """
        

        # load segmentation model
        device  = 'cuda' if torch.cuda.is_available() else 'cpu'
        seg_model_path = './resources/seg_model.pt'
        cls.seg_model = torch.load(seg_model_path).to(device)
        cls.seg_model.eval()

        # load male regression model
        M_model_path = './resources/M_model.pt'
        cls.M_model = torch.load(M_model_path).to(device)
        cls.M_model.eval()

        # load female regression model
        F_model_path = './resources/F_model.pt'
        cls.F_model = torch.load(F_model_path).to(device)
        cls.F_model.eval()

        # Useful to show example_ai_service.py is ready to take GET calls from AiSvcTest
        logging.info("Model Loaded Successfully")
        logging.info("AI Service Initialized Successfully, Ready to Receive AI Service Calls")

    # ---------------------------------------------------------------------------
    def process_study(self):
        """
        Do the actual work to process this study.
        DICOM images that have been filtered and selected are available in
        self.ai_job.selected_series_list.  The files that back these images
        are located in self.ai_job.image_folder.

        Full implementation of this method should include partner logic for
        analyzing a study and calls to upload_document to include analysis
        output in the results reported to AI Marketplace.  See the
        documentation for AIM Service for a full reference of available values.

        Returns
        -------
            None
        """
        logging.info("Beginning process_study job")
        
        

        
        # Capture start time at beginning of this method
        self.datetime_start = datetime.now().isoformat()

        # Path to Dicom folder
        path_to_dicom = self.ai_job.primary_study.folder

        # Form filename header from partner and service name
        self.filename_header = "{PARTNER}-{SERVICE}-".format(
                                        PARTNER=self.partner_name,
                                        SERVICE=self.service_name,
                                        )

        
        logging.info(self.filename_header)

        

        # ------------ Workflow for this particular example ------------
        # Pull image dataset from dicom folder served by PIN
        logging.debug("Running Internal Series Selection")
        image_datasets = series_selection(path_to_dicom)
        logging.debug("Internal Series Selection Complete")


        # Proprocess selected study
        logging.debug("Running Preprocessing")
        
        def preprocess_image(dcm):
            device  = 'cuda' if torch.cuda.is_available() else 'cpu'
            image = read_image(dcm)

            # crop, resize, predict and apply mask
            image = crop_image(image)
            resized = resize_image(image, 256)
            image = intensity_transfer(resized)
            transformed = seg_img_transform(Image.fromarray(image).convert('RGB'), seg_transform, device)
            logging.info('image transformed')
            with torch.no_grad():
                pred_mask = torch.sigmoid(self.seg_model(transformed))
                pred_mask = (pred_mask > 0.5).float().detach().cpu().numpy().squeeze()

            masked = np.ma.masked_where(pred_mask == 0, resized)
            masked = np.ma.MaskedArray.filled(masked,fill_value=0)
            logging.info('image masked')
            
            # post processing
            image = contrast_stretching(masked,70,100)
            image = intensity_transfer2(image, 35,35)
                    
            logging.info('saving image ....')

            # save masked image as png file
            final = Image.fromarray(image).convert('RGB')
            final.save("./image.png", format = 'png')
            logging.info('png saved Successfully')
            


        sex = [dcm[0x0010, 0x0040][:] for dcm in image_datasets][0]
        preprocessed_ims = [preprocess_image(dcm) for dcm in image_datasets]
        len_ims = len(preprocessed_ims)
        logging.debug("Preprocessing Complete")
        logging.debug(f"{len_ims} images preprocessed")

        reg_transform = transforms.Compose([transforms.Resize(225),
                                              transforms.CenterCrop(224),
                                              transforms.ToTensor(),
                                              transforms.Normalize([0.485, 0.456, 0.406],
                                                                  [0.229, 0.224, 0.225])])
        img = './image.png'
        device  = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        def reg_img_transform(img,reg_transform,device):
            image = Image.open(img)
            image = reg_transform(image)
            return image.to(device).unsqueeze(0)
            
        
        if sex == 'M':
            reg_model = self.M_model
        else:
            reg_model = self.F_model
        
        logging.info(f'Sex = {sex}')
        
        
        transformed = reg_img_transform(img,reg_transform, device)

        logging.info('Predicting age ...')

        try:
            with torch.no_grad():
                output = reg_model(transformed)
            error_status = 'OK'
            logging.info(f'Predicted age: {output.item()} years')
        except TypeError as e:
            error_status = e
            logging.error("Bone age failed to generate")

        

        # ------------ Writing FHIR.json results with diagnostic_report from ai_service ------------
        def create_diagnostic_report(prediction, error_rate):

            long_conclusion_text = f"Predicted bone age is: {prediction} years with error margin {error_rate} months."

            self.conclusion = "Bone Age"

            report = DiagnosticReport(
                divtext="<div xmlns=\"http://www.w3.org/1999/xhtml\"><p><b>Example AI model finding:</b></p><p>" +
                long_conclusion_text + "</p></div>",
                conclusion=self.conclusion
            )

            report.add_device(
                self.partner_name + "-" + self.service_name,  # "device_id"
                self.partner_name,  # "manufacturer"
                self.service_name,  # "name"
                self.version,
            )

            # The AI Technique is added as a separate Observation and is used by PowerScribe 360
            report.set_ai_technique("MGB Bone Age")

            study = report.set_study(
                dcm=next(self.ai_job.primary_study.series).image,  # it will open this dcm to get StudyUID and other info
                procedure_code="36563-5",  # Put a LOINC code here https://loinc.org/
                procedure_display="bone age model",
                procedure_text="bone age model")

            test_dcm = next(self.ai_job.primary_study.series).image
            issuerOfPatientID = test_dcm.get("IssuerOfPatientID", 'unknown')

            # Add Observation to be parsable by MGB system
            ob1 = report.add_observation(
                        study,
                        "mgb.nuance.common",
                        "Common MGB tracking codes",
                        "https://datascience.massgeneralbrigham.org/",
                        dcm=None,  # os.path.join(path_to_dicom, self.filename_l), #relevant filepath goes here
                        body_part_code="RID50364",
                        body_part_text="normal anatomy",
                        )

            ob1.component.add_value_string(
                "mgb.nuance.starttime",
                "date and time when inference started",
                "https://datascience.massgeneralbrigham.org/",
                self.datetime_start
            )
            ob1.component.add_value_string(
                "mgb.nuance.endtime",
                "date and time when inference ended",
                "https://datascience.massgeneralbrigham.org/",
                datetime.now().isoformat()
            )
            
            ob1.component.add_value_quantity(
                "mgb.nuance.boneage.predictedage",
                "predicted bone age",
                "https://datascience.massgeneralbrigham.org/", prediction, "years",
                "http://unitsofmeasure.org"
            )
            ob1.component.add_value_quantity(
                "mgb.nuance.boneage.errormargin",
                "error margin",
                "https://datascience.massgeneralbrigham.org/", error_rate, "months",
                "http://unitsofmeasure.org"
            )

            report.set_summary(long_conclusion_text)

            self.report = report
            return long_conclusion_text
            
        error_rate = 9.38 if sex == 'M' else 10.56


        if (error_status == "OK"):
            logging.debug("Processed Images")
            create_diagnostic_report(output.item(),error_rate)
            logging.debug("Created Diagnostic Report")
            self._save_and_upload_diagnostic_report()
            logging.debug("Uploaded Diagnostic Report")
            self._upload_results()
            logging.debug("Uploaded .dcm")
            self.set_transaction_status("ANALYSIS_COMPLETE")
            # Default reason is ANALYSIS_COMPLETE #reason=self.conclusion
        else:
            self._upload_result("results.json", application="application/json")
            raise AiServiceException(error_status)
        logging.info("process_study Complete")


    # ---------------------------------------------------------------------------
    def _save_and_upload_diagnostic_report(self):

        # write the json to a file
        report_filename = (
            self.ai_job.output_folder
            / "{PARTNER}-{SERVICE}-FHIR.json".format(
                PARTNER=self.partner_name, SERVICE=self.service_name
            )
        )
        self.report.write_to_file(report_filename)


        self.upload_document(
            report_filename,
            content_type="application/json",
            document_detail="{PARTNER} {SERVICE} results from {DATE}".format(
                PARTNER=self.partner_name,
                SERVICE=self.service_name,
                DATE=datetime.now(),
                ),
            tracking_uids=None,
            )

    # ---------------------------------------------------------------------------
    def _upload_results(self):

        # find all the files that were generated and upload all of them
        all_dcm_files = glob.glob(os.path.join("/", "*"))

        print("ALL FILES", all_dcm_files)

        for filepath in all_dcm_files:
            fn = os.path.basename(filepath)

            if ("_SR" in fn):
                self._upload_result(filepath, application="application/sr+dicom")
            elif (".json" in fn):
                self._upload_result(filepath, application="application/json")
            elif (".nii" in fn) or (".log" in fn):
                self._upload_result(filepath, application="application/octet-stream")
            elif (".dcm" in fn.lower()):
                self._upload_result(filepath, application="application/dicom")

    # ---------------------------------------------------------------------------
    def _upload_result(self, filepath, application="application/dicom"):

        new_series_uid = related_uid(next(self.ai_job.primary_study.series).uid)

        self.upload_document(
            filepath,
            content_type=application,
            document_detail="{PARTNER} {SERVICE} results from {DATE}".format(
                PARTNER=self.partner_name,
                SERVICE=self.service_name,
                DATE=datetime.now(),
            ),
            series_uid=new_series_uid,
            tracking_uids=None,
            )


# -------------------------------------------------------------------------------
if __name__ == "__main__":
    """
    Note that AiService takes a AI Job Processor class as an initialization
    argument.
    This AI Job Processor is instantiated each time it processes a job.
    This has the advantage of guaranteeing a clean slate, but also means that
    important context must be stored in the class (not the instance)

    Important context includes models and other resources that should be loaded
    only once.
    """

    AiService(BoneageAiService).start()  # Call your AI Service Class with this command
