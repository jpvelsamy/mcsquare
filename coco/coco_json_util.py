import json
from typing import List, Dict, Optional
from coco import ImageInfo, Category, Annotation, CocoInfo, CocoDataset


class CocoUtils:
    @staticmethod
    def unmarshal(json_file: str) -> CocoDataset:
        with open(json_file, 'r') as f:
            data = json.load(f)

        images = [ImageInfo(**image) for image in data['images']]
        categories = [Category(**category) for category in data['categories']]
        annotations = [Annotation(**annotation) for annotation in data['annotations']]
        info = CocoInfo(**data['info'])

        return CocoDataset(images, categories, annotations, info)

    @staticmethod
    def marshal(coco_dataset: CocoDataset, json_file: str):
        data = {
            "images": [vars(image) for image in coco_dataset.images],
            "categories": [vars(category) for category in coco_dataset.categories],
            "annotations": [vars(annotation) for annotation in coco_dataset.annotations],
            "info": vars(coco_dataset.info)
        }
        with open(json_file, 'w') as f:
            json.dump(data, f, indent=4)
