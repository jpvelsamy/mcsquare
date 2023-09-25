import unittest
from coco.coco_json_util import CocoUtils
import os
import logging

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('testing')


class MyTestCase(unittest.TestCase):

    def setUp(self):
        current_file_path: str = os.path.abspath(__file__)
        current_directory: str = os.path.dirname(current_file_path)
        self.parent_directory: str = os.path.dirname(current_directory)

        self.ann_file: str = self.parent_directory + "/annotations/coco/coco_label_studio_chkbox_detect_60_62/result-sample.json"
        if not os.path.exists(self.ann_file):
            raise FileNotFoundError(f"The file '{self.ann_file}' does not exist.")

    def test_coco_unmarshall(self):
        # Unmarshal JSON to CocoDataset
        coco_dataset = CocoUtils.unmarshal(self.ann_file)
        logger.info("file content %s", coco_dataset.to_json())

        self.assertIsNotNone(coco_dataset)


if __name__ == '__main__':
    unittest.main()
