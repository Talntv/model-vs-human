import os
import sys
from PIL import Image as im
from modelvshuman.models.pytorch.model_zoo import clip, resnet50_clip_soft_labels, resnet50_trained_on_SIN
from modelvshuman.datasets.decision_mappings import ImageNetProbabilitiesTo16ClassesMapping
import torch
from PIL import Image
import torchvision.transforms as transforms
from  torch.nn.parallel.data_parallel import DataParallel

def drop_last_layer(data_parrallel_model: DataParallel):
    """
    Remove last layer of the model. We first have to extract the actual layers of the model from the wrapping libraries, and then drop the last layer, FC (=fully connected -> 2048 down to 1000 features of probabilities)
    """
    model = data_parrallel_model.model._modules['module']
    newmodel = torch.nn.Sequential(*list(model.children())[:-1])
    newmodel.eval() # evaluate mode, don't calculate the deriatives(gradients)
    return newmodel


current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, '..'))


# Read model
model = clip("clip")
feature_extractor_model = drop_last_layer(model)

# Read the image
normal_cat = Image.open(os.path.join(current_dir, 'cat7_original.png'))
elephant_texture_cat = Image.open(os.path.join(current_dir, 'cat7-elephant1.png'))
elephant_texture_elephant = Image.open(os.path.join(current_dir, 'elephant5-elephant1.png'))
another_normal_cat = Image.open(os.path.join(current_dir, 'cat3-cat1.png'))
# Define a transform to convert the image to a PyTorch tensor
normal_cat_tensor = transforms.ToTensor()(normal_cat).unsqueeze(0) # the tensor expects 4D, drop the 1st dimension if we only use one picture
elephant_texture_cat_tensor = transforms.ToTensor()(elephant_texture_cat).unsqueeze(0)
elephant_texture_elephant_tensor = transforms.ToTensor()(elephant_texture_elephant).unsqueeze(0)
another_normal_cat_tensor = transforms.ToTensor()(another_normal_cat).unsqueeze(0)


output_numpy_normal_cat = feature_extractor_model.forward(normal_cat_tensor)
output_numpy_elephant_cat = feature_extractor_model.forward(elephant_texture_cat_tensor)
output_numpy_elephant_elephant = feature_extractor_model.forward(elephant_texture_elephant_tensor)
output_numpy_another_cat =feature_extractor_model.forward(another_normal_cat_tensor)


# torch.cdist(output_numpy_pic1,output_numpy_pic2)**2
cat_from_another_cat = sum(((output_numpy_normal_cat - output_numpy_another_cat)**2).reshape(2048))
output_numpy_another_cat
cat_from_elephant_cat = sum(((output_numpy_normal_cat - output_numpy_elephant_cat)**2).reshape(2048))
cat_from_elephant_elephant = sum(((output_numpy_normal_cat - output_numpy_elephant_elephant)**2).reshape(2048))
elephant_cat_from_elephant_elephant = sum(((output_numpy_elephant_cat - output_numpy_elephant_elephant)**2).reshape(2048))
# last_layer_img = im.fromarray(output_numpy)
# last_layer_img.save('gfg_dummy_pic.png')


# maper = ImageNetProbabilitiesTo16ClassesMapping()
# print(maper(output_numpy)[0][0])
print("--Finish--")
