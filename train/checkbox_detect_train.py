from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torch.utils.data import DataLoader
import torch

import json
import os
from torchvision.transforms import ToTensor

from train.data_set import CheckboxAnnotation


def move_targets_to_device(targets, device):
    new_targets = []  # Create a new list for the modified targets
    for t in targets:  # Iterate over each target dictionary in the list
        new_t = {}  # Create a new dictionary for the target
        new_targets.append(t)  # Add the new dictionary to the list
    return new_targets


# Now you can call this function in your training loop:


def get_dataset(image_dir: str, annotations_file: str, transform: ToTensor):
    # Load COCO data
    with open(annotations_file, 'r') as f:
        coco_data = json.load(f)

    # Create a list of image paths
    image_paths = [os.path.join(image_dir, image['file_name']) for image in coco_data['images']]

    # Convert COCO annotations to Faster R-CNN format
    annotations = []
    for annotation in coco_data['annotations']:
        # Convert bounding box coordinates
        x, y, width, height = annotation['bbox']
        x1, y1, x2, y2 = x, y, x + width, y + height

        # Create a new annotation
        new_annotation = {
            'boxes': torch.tensor([[x1, y1, x2, y2]]),  # Wrap the box coordinates in a list to create a 2D tensor
            'labels': torch.tensor([annotation['category_id']]),  # Wrap the category ID in a list to create a 1D tensor
        }

        annotations.append(new_annotation)

    # Create the dataset
    dataset = CheckboxAnnotation(image_paths, annotations, transform)

    return dataset


def collate_fn(batch):
    images, targets = zip(*batch)  # Unzip the batch
    return list(images), list(targets)  # Create lists of images and targets


# Initialize the dataloader with the custom collate function


# Assuming you have your image paths and annotations

current_file_path: str = os.path.abspath(__file__)
current_directory: str = os.path.dirname(current_file_path)
parent_directory: str = os.path.dirname(current_directory)
img_folder: str = parent_directory + "/annotations/coco/coco_label_studio_chkbox_detect_60_62/images"
if not os.path.exists(img_folder):
    raise FileNotFoundError(f"The file '{img_folder}' does not exist.")

ann_file: str = parent_directory + "/annotations/coco/coco_label_studio_chkbox_detect_60_62/result.json"
if not os.path.exists(ann_file):
    raise FileNotFoundError(f"The file '{ann_file}' does not exist.")

# Initialize the dataset and dataloader
dataset = get_dataset(img_folder, ann_file, transform=ToTensor())
max_label = max(annotation['labels'].max().item() for annotation in dataset.annotations)
print(f"The maximum label in the dataset is: {max_label}")

# data_loader = DataLoader(dataset, batch_size=2, shuffle=True)
data_loader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)
# Define the device
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# Load a pre-trained model
model = fasterrcnn_resnet50_fpn(pretrained=True)

# replace the classifier with a new one, that has num_classes which is user-defined
num_classes = 5  # 2 classes (person and background)

# get number of input features for the classifier
in_features = model.roi_heads.box_predictor.cls_score.in_features

# replace the pre-trained head with a new one
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

# move model to the right device
model.to(device)

# construct an optimizer
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)

# and a learning rate scheduler
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

# let's train it for 10 epochs
num_epochs = 10

for epoch in range(num_epochs):
    model.train()
    for images, targets in data_loader:
        print("Images type:", type(images))  # Debug line
        print("Targets type:", type(targets))  # Debug line
        print("Targets value:", targets)
        print("Targets length:", len(targets))
        t_images = list(image.to(device) for image in images)
        t_targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        loss_dict = model(t_images, t_targets)

        losses = sum(loss for loss in loss_dict.values())
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

    # update the learning rate
    lr_scheduler.step()

# save model
torch.save(model.state_dict(), 'model_weights.pth')
