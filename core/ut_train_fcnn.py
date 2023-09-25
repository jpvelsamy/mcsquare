import os.path
import torch
from torch.utils.data import random_split, DataLoader
from torchvision.datasets import CocoDetection
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

# Define device
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

current_file_path = os.path.abspath(__file__)
current_directory = os.path.dirname(current_file_path)
parent_directory = os.path.dirname(current_directory)
# Load the dataset
img_folder = parent_directory+"/annotations/coco/coco_label_studio_chkbox_detect_60_62/images"
ann_file = parent_directory+"/annotations/coco/coco_label_studio_chkbox_detect_60_62/result.json"
dataset = CocoDetection(img_folder, ann_file)

# Split the dataset
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# Define the data loaders
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=4)

# Load the pretrained model
model = fasterrcnn_resnet50_fpn(pretrained=True)

# Get the number of input features for the classifier
in_features = model.roi_heads.box_predictor.cls_score.in_features

# Define the number of output classes (including the background)
num_classes = 3  # e.g., background, checkbox, text

# Replace the pre-trained head with a new one
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

# Move the model to the right device
model.to(device)

# Define the optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0005)

# Number of training epochs
num_epochs = 10

# Training loop
for epoch in range(num_epochs):
    for images, targets in train_loader:
        # Move the images and targets to the right device
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # Forward pass
        loss_dict = model(images, targets)

        # Calculate the total loss
        losses = sum(loss for loss in loss_dict.values())

        # Backward pass and optimization
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

# Save the trained model
torch.save(model.state_dict(), current_directory+"/fine_tuned_model.pth")
