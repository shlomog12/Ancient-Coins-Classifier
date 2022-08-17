import torch
from torchvision.transforms.transforms import ToTensor
from torch import nn
from torchvision import models
import numpy as np
import pytz
from datetime import datetime
from PIL import Image
import os
import glob

def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


def initialize_model(model_name, num_classes, feature_extract, use_pretrained=True):

    model_ft = None
    input_size = 0

    if model_name == "resnet":
        """ Resnet18
        """
        model_ft = models.resnet18(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224


    elif model_name == "vgg":
        """ VGG11_bn
        """
        model_ft = models.vgg11_bn(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        input_size = 224


    else:
        print("Invalid model name, exiting...")
        exit()

    return model_ft, input_size


def get_inference(img, model_path='../models/model_ft.pth'):
    tensor_img = get_tensor_of_image(img)
    print(tensor_img.shape)

    num_classes = 3
    feature_extract = False
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, input_size = initialize_model('resnet', num_classes, feature_extract, use_pretrained=False)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    list_of_x = [tensor_img]
    inputs = torch.stack(list_of_x)
    inputs = inputs.to(device)

    with torch.set_grad_enabled(False):
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        print('res:')
        print(preds.tolist())
        print('___________')
        return preds.tolist()[0]


def get_tensor_of_image(image):
    transform = ToTensor()
    img = transform(image)
    return img


def normalization( proba_pred_y):
    res =[]
    t = proba_pred_y[0]
    for i in t:
        x = np.exp(i)
        res.append(x)
    return res



def resize_image(file):
    timezone = pytz.timezone("Israel")
    now_time = datetime.now(timezone).strftime("%d-%m-%Y_%H-%M-%S")
    current_name = f'data/{now_time}.jpg'
    file.save(current_name)
    img = Image.open(current_name)
    img = img.resize((224, 224), Image.ANTIALIAS)
    return img


def remove_cache():
    files = glob.glob('data/*.jpg')
    for f in files:
        os.remove(f)
