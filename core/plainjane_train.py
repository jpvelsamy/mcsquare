import logging
import os
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.datasets import CocoDetection
from torch.utils.data import DataLoader
from core.custom_dataset import CheckboxCocoDetection

# Load a pre-trained Faster R-CNN model
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

# Change the classifier to a new one, num_classes includes the background class
num_classes = 3  # 1 class (your new class) + background
# get the number of input features for the classifier
in_features = model.roi_heads.box_predictor.cls_score.in_features
# replace the pre-trained head with a new one
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

# Define the dataset
current_file_path: str = os.path.abspath(__file__)
current_directory: str = os.path.dirname(current_file_path)
parent_directory:str = os.path.dirname(current_directory)
img_folder: str = parent_directory + "/annotations/coco/coco_label_studio_chkbox_detect_60_62"
ann_file: str = parent_directory + "/annotations/coco/coco_label_studio_chkbox_detect_60_62/result.json"

dataset = CheckboxCocoDetection(img_folder, ann_file)

# Define the dataloader
data_loader = DataLoader(dataset, batch_size=2, shuffle=True)

# Move model to gpu if available
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)

# Define the optimizer
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    for images, targets in data_loader:
        images = list(image.to(device) for image in images)
        print(targets)
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

# Save the model
torch.save(model.state_dict(), "model.pth")
