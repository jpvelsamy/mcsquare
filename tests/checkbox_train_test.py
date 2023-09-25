from typing import *
import unittest
import os
from typing import Any

import torch
import torchvision
from core.checkbox_train_fcnn import CheckboxDetector  # assuming your CheckboxDetector class is in checkbox_detector.py
import logging
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('testing')
class TestCheckboxDetector(unittest.TestCase):
    def setUp(self):
        current_file_path: str = os.path.abspath(__file__)
        current_directory: str = os.path.dirname(current_file_path)
        self.parent_directory:str = os.path.dirname(current_directory)
        self.img_folder: str = self.parent_directory + "/annotations/coco/coco_label_studio_chkbox_detect_60_62/"
        if not os.path.exists(self.img_folder):
            raise FileNotFoundError(f"The file '{self.img_folder}' does not exist.")

        self.ann_file: str = self.parent_directory+"/annotations/coco/coco_label_studio_chkbox_detect_60_62/result.json"
        if not os.path.exists(self.ann_file):
            raise FileNotFoundError(f"The file '{self.ann_file}' does not exist.")

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.num_classes = 3
        if logger.isEnabledFor(logging.INFO):
            logger.info("Parent directory: %s, Target device: %s, feature classes: %s", self.parent_directory, self.device, self.num_classes)
        self.detector = CheckboxDetector(self.img_folder, self.ann_file, self.device, self.num_classes)

    def test_initialize(self):
        if logger.isEnabledFor(logging.INFO):
            logger.info("Initializing model")
        self.detector._initialize_model(3)
    def test_train(self):
        try:
            if logger.isEnabledFor(logging.INFO):
                logger.info("Py torch version: %s", torch.__version__)
                logger.info("Torch model version: %s", torchvision.__version__)
                logger.info("Source directory:%s "+self.parent_directory)
                logger.info("Image folder:%s " + self.img_folder)
                logger.info("Annotation folder:%s " + self.ann_file)
            self.detector.train(1)  # train for 1 epoch for testing
        except Exception as e:
            self.fail(f"Training failed with {e}")

    def test_save_model(self):
        try:
            self.detector.save_model(self.parent_directory+"/fine_tuned_model.pth")
            # Check if the model file is saved
            self.assertTrue(os.path.exists(self.parent_directory+"/fine_tuned_model.pth"))
            # Clean up the model file after testing
            os.remove("fine_tuned_model.pth")
        except Exception as e:
            self.fail(f"Saving model failed with {e}")

if __name__ == '__main__':
    unittest.main()
