from torchvision.transforms.functional import to_tensor
from torchvision.datasets import CocoDetection
from torchvision.transforms import ToTensor
from torch.utils.data import Dataset
import torch


class CheckboxCocoDetection(Dataset):
        def __init__(self, img_folder, ann_file):
            self.coco = CocoDetection(img_folder, ann_file)
            self.transform = ToTensor()

        def __getitem__(self, index):
            img, targets = self.coco[index]
            img = self.transform(img)

            new_targets = []
            for target in targets:
                bbox = target["bbox"]
                x, y, width, height = bbox
                x1, y1, x2, y2 = x, y, x + width, y + height
                new_target = target.copy()  # copy the target to preserve other fields
                new_target["boxes"] = torch.tensor([x1, y1, x2, y2])  # convert to tensor
                del new_target["bbox"]
                new_targets.append(new_target)

            return img, new_targets

        def __len__(self):
            return len(self.coco)

