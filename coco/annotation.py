from typing import List
class Annotation:
    def __init__(self, id: int, image_id: int, category_id: int, segmentation:List[str], bbox: List[float], ignore: int, iscrowd: int,
                 area: float):
        self.id = id
        self.image_id = image_id
        self.category_id = category_id
        self.segmentation = segmentation
        self.bbox = bbox
        self.ignore = ignore
        self.iscrowd = iscrowd
        self.area = area

