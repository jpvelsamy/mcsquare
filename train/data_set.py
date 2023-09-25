from typing import List, Dict, Optional
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor


class CheckboxAnnotation(Dataset):
    def __init__(self, image_paths: List[str], annotations: List[Dict[str, torch.Tensor]],
                 transform: Optional[ToTensor] = None):
        self.image_paths: List[str] = image_paths
        self.annotations: List[Dict[str, torch.Tensor]] = annotations
        self.transform: Optional[ToTensor] = transform

    def __getitem__(self, idx: int) -> tuple:
        img: Image.Image = Image.open(self.image_paths[idx]).convert("RGB")
        if self.transform:
            img: torch.Tensor = self.transform(img)

        boxes: torch.Tensor = self.annotations[idx]["boxes"]
        labels: torch.Tensor = self.annotations[idx]["labels"]

        # Debug lines
        print("Boxes type:", type(boxes))
        print("Boxes value:", boxes)
        print("Labels type:", type(labels))
        print("Labels value:", labels)

        # Ensure boxes and labels are torch tensors
        if not isinstance(boxes, torch.Tensor):
            boxes = torch.tensor(boxes)
        if not isinstance(labels, torch.Tensor):
            labels = torch.tensor(labels)

        # Ensure boxes are float and labels are long int
        boxes = boxes.float()
        labels = labels.long()

        target: Dict[str, torch.Tensor] = {"boxes": boxes, "labels": labels}

        return img, target

    def __len__(self) -> int:
        return len(self.image_paths)
