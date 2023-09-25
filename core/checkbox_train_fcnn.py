from typing import List, Dict
import torch

from torch.utils.data import random_split, DataLoader, Dataset, Subset
from torchvision.datasets import CocoDetection
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor, FasterRCNN_ResNet50_FPN_Weights



class CheckboxDetector:
    def __init__(self, img_folder: str, ann_file: str, device: str, num_classes: int):
        self.device: torch.device = torch.device(device)
        self.dataset: Dataset = CocoDetection(img_folder, ann_file)
        self.model: torch.nn.Module = self._initialize_model(num_classes)

    def _initialize_model(self, num_classes: int) -> torch.nn.Module:
        model: torch.nn.Module = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
        in_features: int = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        model.to(self.device)
        return model

    def get_data_loaders(self, train_ratio: float = 0.8, batch_size: int = 4) -> Dict[str, DataLoader]:
        train_size: int = int(train_ratio * len(self.dataset))
        val_size: int = len(self.dataset) - train_size
        rand_generator = torch.Generator().manual_seed(42)
        splits: List[Dataset] = random_split(self.dataset, [train_size, val_size], generator=rand_generator)
        train_dataset: Dataset = splits[0]
        val_dataset: Dataset = splits[1]

        train_loader: DataLoader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader: DataLoader = DataLoader(val_dataset, batch_size=batch_size)
        return {"train": train_loader, "val": val_loader}

    def train(self, num_epochs: int, lr: float = 0.005):
        data_loaders: Dict[str, DataLoader] = self.get_data_loaders()
        optimizer: torch.optim.Optimizer = torch.optim.SGD(self.model.parameters(), lr=lr, momentum=0.9,
                                                           weight_decay=0.0005)
        train_data_loader: DataLoader = data_loaders["train"]
        data_iter = iter(data_loaders["train"].dataset)
        for epoch in range(num_epochs):
            for images, targets in data_loaders["train"].dataset:
                loss_dict: Dict[str, torch.Tensor] = self.model(images, targets)
                losses: torch.Tensor = sum(loss for loss in loss_dict.values())
                optimizer.zero_grad()
                losses.backward()
                optimizer.step()

    def save_model(self, path: str):
        torch.save(self.model.state_dict(), path)
