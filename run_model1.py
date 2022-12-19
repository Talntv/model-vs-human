import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, '..'))

from modelvshuman.models.pytorch.model_zoo import resnet50_trained_on_SIN
from modelvshuman.datasets.decision_mappings import ImageNetProbabilitiesTo16ClassesMapping
import torch
from PIL import Image
import torchvision.transforms as transforms

model = resnet50_trained_on_SIN("resnet50_trained_on_SIN")

# Read the image
image = Image.open(os.path.join(current_dir, 'b.jpg'))
# Define a transform to convert the image to tensor
transform = transforms.ToTensor()

# Convert the image to PyTorch tensor
tensor = transform(image).unsqueeze(0)

# print the converted image tensor

output_numpy = model.forward_batch(tensor)
maper = ImageNetProbabilitiesTo16ClassesMapping()
print(maper(output_numpy)[0][0])
print("--Finish--")
