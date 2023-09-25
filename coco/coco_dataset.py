import typing
from coco import ImageInfo, Category, Annotation, CocoInfo
import json


class CocoDataset:
    def __init__(self, images: typing.List[ImageInfo], categories: typing.List[Category],
                 annotations: typing.List[Annotation],
                 info: CocoInfo):
        self.images = images
        self.categories = categories
        self.annotations = annotations
        self.info = info

    def to_json(self) -> str:
        data = {
            'images': [vars(image) for image in self.images],
            'categories': [vars(category) for category in self.categories],
            'annotations': [vars(annotation) for annotation in self.annotations],
            'info': vars(self.info)
        }
        return json.dumps(data, indent=4)
