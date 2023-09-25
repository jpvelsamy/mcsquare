import os.path

import cv2
import pytesseract
from matplotlib import pyplot as plt
from torchvision import transforms
from torchvision.models.detection import fasterrcnn_resnet50_fpn
import torch
import torchvision
import sys

checkbox_detection_model = fasterrcnn_resnet50_fpn(pretrained=True)
checkbox_detection_model.eval()
# Load the OCR model (Tesseract)
pytesseract.pytesseract.tesseract_cmd = r'/usr/bin/tesseract'

def print_version(name):
    # Use a breakpoint in the code line below to debug your script.
    #print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.
    print("Python version (major):", sys.version_info.major)
    print("Python version (minor):", sys.version_info.minor)
    print("Py torch version", torch.__version__)
    print("Torch model version", torchvision.__version__)


def load_file(file_name):
    current_file_path = os.path.abspath(__file__)
    current_directory = os.path.dirname(current_file_path)
    # parent_directory = os.path.dirname(current_directory)
    # image_file_path = os.path.join(current_directory, "/samples/humana_samples/output-img/Aaron_Henry_292.jpg")
    image_file_path = current_directory + "/samples/"+file_name
    image = cv2.imread(image_file_path)
    image

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_version('fcnn_eval')
    current_file_path = os.path.abspath(__file__)
    current_directory = os.path.dirname(current_file_path)
    parent_directory = os.path.dirname(current_directory)
    image_file_path = parent_directory + "/samples/Aaron_Henry_292.jpg"
    image = cv2.imread(image_file_path)
    image_grayscaled = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    transform = transforms.Compose([transforms.ToTensor()])
    image_transform = transform(image_grayscaled)
    torch.jit.is_scripting()
    predictions = checkbox_detection_model([image_transform])
    checkbox_boxes = predictions[0]['boxes'].tolist()
    checkbox_scores = predictions[0]['scores'].tolist()
    # Extract checkbox label using OCR
    for box, score in zip(checkbox_boxes, checkbox_scores):
        if score > 0.5:  # Adjust the score threshold as per your requirement
            x1, y1, x2, y2 = box
            checkbox_region = image[y1:y2, x1:x2]
            label_text = pytesseract.image_to_string(checkbox_region)
            print("Checkbox Label:", label_text)

    # Visualize the checkbox detection results
    image = cv2.cvtColor(image_transform.numpy(), cv2.COLOR_RGB2BGR)
    for box in checkbox_boxes:
        x1, y1, x2, y2 = box
        cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
    plt.imshow(image)
    plt.axis('off')
    plt.show()


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
