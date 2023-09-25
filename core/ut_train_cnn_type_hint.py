from typing import List, Dict, Any
import os.path
import torch

from torch.utils.data import random_split, DataLoader, Dataset, Subset
from torchvision.datasets import CocoDetection
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

# Define device
device: torch.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# Load the dataset
current_file_path = os.path.abspath(__file__)
current_directory = os.path.dirname(current_file_path)
image_file_path = current_directory + "/samples/Aaron_Henry_292.jpg"
# Load the dataset
img_folder = current_directory+"/annotations/coco/coco_label_studio_chkbox_detect_60_62/images"
ann_file = current_directory+"/annotations/coco/coco_label_studio_chkbox_detect_60_62/result.json"
dataset: Dataset = CocoDetection(img_folder, ann_file)

# Split the dataset
train_size: int = int(0.8 * len(dataset))
val_size: int = len(dataset) - train_size
generator1 = torch.Generator().manual_seed(42)
splits: List[Subset] = random_split(dataset, [train_size, val_size],  generator=generator1)

train_dataset: Subset = splits[0]
val_dataset: Subset = splits[1]



# Define the data loaders
train_loader: DataLoader = DataLoader(train_dataset, batch_size=4, shuffle=True)

# Load the pretrained model
model: torch.nn.Module = fasterrcnn_resnet50_fpn(pretrained=True)

# Get the number of input features for the classifier
in_features: int = model.roi_heads.box_predictor.cls_score.in_features

# Define the number of output classes (including the background)
num_classes: int = 3  # e.g., background, checkbox, text

# Replace the pre-trained head with a new one
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

# Move the model to the right device
model.to(device)

# Define the optimizer
optimizer: torch.optim.Optimizer = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0005)

# Number of training epochs
num_epochs: int = 10

# Training loop
for epoch in range(num_epochs):
    for images, targets in train_loader:
        # Move the images and targets to the right device
        images: List[torch.Tensor] = list(image.to(device) for image in images)
        targets: List[Dict[str, torch.Tensor]] = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # Forward pass
        loss_dict: Dict[str, torch.Tensor] = model(images, targets)

        # Calculate the total loss
        losses: torch.Tensor = sum(loss for loss in loss_dict.values())

        # Backward pass and optimization
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

# Save the trained model
torch.save(model.state_dict(), "fine_tuned_model.pth")
