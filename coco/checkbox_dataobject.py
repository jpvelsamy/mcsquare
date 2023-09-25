from tkinter import Image
from typing import List, Dict, Optional, Tuple
from torchvision.datasets import CocoDetection
from torchvision.transforms import ToTensor
from torch.utils.data import Dataset
import torch
from coco import ImageInfo, Category, Annotation, CocoInfo, CocoDataset
from PIL import Image


class CheckboxDataset(Dataset):
    def __init__(self, coco_data: CocoDataset, transform: Optional[ToTensor] = None):
        self.image_paths: List[str] = [img_info.file_name for img_info in coco_data.images]
        self.annotations: List[Annotation] = coco_data.annotations
        self.transform: Optional[ToTensor] = transform

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        raw_image: Image = Image.open(self.image_paths[idx]).convert("RGB")

        # Check if transform is provided and apply
        if self.transform:
            transformed_image: torch.Tensor = self.transform(raw_image)
        else:
            transformed_image: torch.Tensor = ToTensor()(raw_image)

        raw_boxes: List[float] = self.annotations[idx].bbox
        boxes_tensor: torch.Tensor = torch.tensor(raw_boxes, dtype=torch.float32)

        raw_labels: int = self.annotations[idx].category_id
        labels_tensor: torch.Tensor = torch.tensor([raw_labels], dtype=torch.int64)

        target: Dict[str, torch.Tensor] = {"boxes": boxes_tensor, "labels": labels_tensor}

        return transformed_image, target

    def __len__(self) -> int:
        return len(self.image_paths)
