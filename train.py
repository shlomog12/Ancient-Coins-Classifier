
from matplotlib import pyplot as plt
from torchvision.io import read_image
import torchvision.transforms as transforms
import PIL
from torchvision import datasets

import torch.optim as optim
import torch
import time, copy
from torchvision.transforms.transforms import ToTensor
from torch.utils.data import DataLoader
from torch import nn
from torchvision import models
from Data import CoinImageDataset
from config import Config


training_data = Config.training_data
training_annotations = Config.training_annotations
test_data = Config.test_data
test_annotations = Config.training_annotations

dataset_train = CoinImageDataset(annotations_file=training_annotations, img_dir=training_data, transform=ToTensor())
dataloader_train = DataLoader(dataset_train, batch_size = 10, shuffle=True)

dataset_test = CoinImageDataset(annotations_file=training_annotations, img_dir=training_data, transform=ToTensor())
dataloader_test = DataLoader(dataset_test, batch_size = 10, shuffle=True)



# batch = next(iter(dataloader_train))
# plt.imshow(batch["image"][0].permute(2,1,0))
# plt.show()


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


def train_model(model, dataloaders, criterion, optimizer, num_epochs=25, is_inception=False):
    since = time.time()

    val_acc_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for sample in dataloaders[phase]:
                inputs = sample['image']
                labels = sample['label']
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):

                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'val':
                val_acc_history.append(epoch_acc)

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    model.load_state_dict(best_model_wts)
    return model, val_acc_history

def get_learnable_parmas(model_ft, feature_extract):
    params_to_update = model_ft.parameters()
    if feature_extract:
        params_to_update = []
        for name, param in model_ft.named_parameters():
            if param.requires_grad == True:
                params_to_update.append(param)
                print("\t",name)
    else:
        for name, param in model_ft.named_parameters():
            if param.requires_grad == True:
                print("\t",name)
    return  params_to_update


device = 'cuda' if torch.cuda.is_available() else 'cpu'
num_classes = 2
num_epochs = 15
feature_extract = True
model_ft, input_size = initialize_model('resnet', num_classes, feature_extract,use_pretrained=True)
criterion = nn.CrossEntropyLoss()


model_ft = model_ft.to(device)


print("Params to learn:")



dataloaders_dict = {'train': dataloader_train, 'val': dataloader_test}
params_to_update = get_learnable_parmas(model_ft,feature_extract)
optimizer_ft = optim.SGD(params_to_update, lr=0.001, momentum=0.9)

model_ft, hist = train_model(model_ft, dataloaders_dict, criterion, optimizer_ft, num_epochs=num_epochs)