# -*- coding: utf-8 -*-
"""M22CS061_DLOps_Assignment-3_Q2.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1sCullqTVrqh-QGT8v6jMGnrarx-cgf27
"""

!pip install -q onnx
!pip install -q onnxruntime
!pip install -q torchmetrics
!pip install -q torch-tb-profiler

"""## Import Libraries"""

import torch
from torch import nn

import torchvision
from torchvision import models, transforms
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from tqdm.auto import tqdm
from pathlib import Path
import gc

from torchmetrics.classification import MulticlassAccuracy

import torch.profiler
import torch.nn.functional as F
from timeit import default_timer as timer
from copy import deepcopy
import torch.onnx
import onnxruntime
from onnxruntime.quantization import quantize_dynamic, QuantType
import onnx
import os

"""## Device agnostic code"""

device = 'cuda' if torch.cuda.is_available() else 'cpu'
device

"""# Load data

### Preprocess and augment the data
"""

train_transform = transforms.Compose([
                                transforms.AutoAugment(transforms.autoaugment.AutoAugmentPolicy.CIFAR10),
                                transforms.ToTensor(),
                                # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                              ])

test_transform = transforms.Compose([
                                transforms.ToTensor(),
                                # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                              ])

train_data = datasets.CIFAR10(root='/data', train = True, transform = train_transform, download = True)
test_data = datasets.CIFAR10(root='/data', train = False, transform = test_transform, download = True)

len(train_data), len(test_data)

"""## Make DataLoader"""

BATCH_SIZE = 32

train_dataloader = DataLoader(dataset = train_data, batch_size = BATCH_SIZE, shuffle = True)
test_dataloader = DataLoader(dataset = test_data, batch_size = BATCH_SIZE, shuffle = False)

print(f"we have total {len(train_dataloader)} of each {BATCH_SIZE} batch")
print(f"we have total {len(test_dataloader)} of each {BATCH_SIZE} batch")

plt.imshow(train_data[0][0].permute(1,2,0))
plt.title(train_data[0][1])

class_name = train_data.classes
class_name

class_to_idx = train_data.class_to_idx
class_to_idx

num_class = len(class_name)

torch.manual_seed(64)
plt.figure(figsize=(10,10))
row, col = 3, 3

for i in range(1, row*col+1):
  random_idx = torch.randint(0, len(train_data), size = [1]).item()
  image, label = train_data[random_idx]
  plt.subplot(row, col, i)
  # plt.imshow(image.squeeze(), cmap = 'gray')
  plt.imshow(image.permute(1,2,0))
  plt.title(class_name[label])
  plt.axis(False)

"""## Train function for tensor board"""

def train(data, model: torch.nn.Module, loss_fn, optimizer):
    inputs, labels = data[0].to(device=device), data[1].to(device=device)
    outputs = model(inputs)
    loss = loss_fn(outputs, labels)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

"""## train and test loop"""

# train
def train_loop(model: torch.nn.Module, dataloader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module, optimizer: torch.optim.Optimizer,
               accuracy_fn, device: torch.device = device):
  
  model.train()
  train_loss, train_acc = 0, 0
  for batch, (x_train, y_train) in enumerate(dataloader):

    if device == 'cuda':
      x_train, y_train = x_train.to(device), y_train.to(device)

    # 1. Forward step
    pred = model(x_train)

    # 2. Loss
    loss = loss_fn(pred, y_train)

    # 3. Grad zerostep
    optimizer.zero_grad()

    # 4. Backward
    loss.backward()

    # 5. Optimizer step
    optimizer.step()

    # print(torch.argmax(pred, dim=1))
    
    # print(y_train)

    # print(torch.argmax(pred, dim=0))
    acc = accuracy_fn(y_train, torch.argmax(pred, dim=1))
    train_loss += loss
    train_acc += acc

  train_loss /= len(dataloader)
  train_acc /= len(dataloader)

  return train_loss, train_acc


# test
def test_loop(model: torch.nn.Module, dataloader: torch.utils.data.DataLoader,
              loss_fn: torch.nn.Module, accuracy_fn, 
              device: torch.device = device):
  
  
  test_loss, test_acc = 0, 0

  model.eval()
  with torch.inference_mode():
    for x_test, y_test in dataloader:

      if device == 'cuda':
        x_test, y_test = x_test.to(device), y_test.to(device)

      # 1. Forward
      test_pred = model(x_test)
      
      # 2. Loss and accuray
      test_loss += loss_fn(test_pred, y_test)
      test_acc += accuracy_fn(y_test, torch.argmax(test_pred, dim=1))

    test_loss /= len(dataloader)
    test_acc /= len(dataloader)

  return test_loss, test_acc

"""## Plot Function for Loss/Accuracy VS Epoches"""

def plot_graph(train_losses, test_losses, train_accs, test_accs):
  plt.figure(figsize = (20, 8))
  plt.subplot(1, 2, 1)
  plt.plot(range(len(train_losses)), train_losses, label = "Train Loss")
  plt.plot(range(len(test_losses)), test_losses, label = "Test Loss")
  plt.legend()
  plt.xlabel("Epoches")
  plt.ylabel("Loss")
  # plt.show()

  plt.subplot(1, 2, 2)
  plt.plot(range(len(train_accs)), train_accs, label = "Train Accuracy")
  plt.plot(range(len(test_accs)), test_accs, label = "Test Accuracy")
  plt.legend()
  plt.xlabel("Epoches")
  plt.ylabel("Accuracy")
  plt.show()

"""## Loss and Accuracy Function"""

loss_fn = nn.CrossEntropyLoss()

accuracy_fn = MulticlassAccuracy(num_classes = len(class_name)).to(device)

"""## Save Model"""

def save_model(MODEL_NAME, model: torch.nn.Module):

  MODEL_PATH = Path("drive/MyDrive/Course/Sem2/DLOps/")
  MODEL_PATH.mkdir(parents = True, exist_ok = True)

  # MODEL_NAME = "ad_model.pth"
  MODEL_PATH_SAVE = MODEL_PATH / MODEL_NAME

  print(f"model saved at: {MODEL_PATH_SAVE}")
  torch.save(obj = model.state_dict(), f = MODEL_PATH_SAVE)

"""## Build Model

### custom architecture: - conv -> conv -> maxpool(2,2) -> conv -> maxpool(2,2) -> conv -> maxpool(2,2)
"""

class CustomCNN(nn.Module):
  def __init__(self, in_channel: int, out_channel: int, out_feature: int):
    super().__init__()
    self.conv_block = nn.Sequential(
                                        nn.Conv2d(in_channels = in_channel, out_channels = out_channel, kernel_size = 3, stride = 1),
                                        nn.ReLU(),
                                        nn.Conv2d(in_channels = out_channel, out_channels = out_channel, kernel_size = 3, stride = 1),
                                        nn.ReLU(),
                                        nn.MaxPool2d(kernel_size = 2, stride = 2),
                                        nn.Conv2d(in_channels = out_channel, out_channels = out_channel, kernel_size = 3, stride = 1),
                                        nn.ReLU(),
                                        nn.MaxPool2d(kernel_size = 2, stride = 2),
                                        nn.Conv2d(in_channels = out_channel, out_channels = out_channel, kernel_size = 3, stride = 1),
                                        nn.ReLU(),
                                        nn.MaxPool2d(kernel_size = 2, stride = 2)
                                  )
    self.linear_block = nn.Sequential(
                                        nn.Flatten(),
                                        nn.Linear(in_features = 20*2*2, out_features = 1024),
                                        nn.ReLU(),
                                        nn.Linear(in_features = 1024, out_features = out_feature)                                        
                                    )
    
  def forward(self, x):
    conv_x = self.conv_block(x)
    # print(f"conv_x shape: {conv_x.shape}")
    out = self.linear_block(conv_x)

    return out

custom_model = CustomCNN(3, 20, num_class).to(device)
custom_model

"""#### train model"""

# init. epochs
epoches = 11

custom_model = CustomCNN(3, 20, num_class).to(device)

# optimizer
optimizer = torch.optim.Adam(params = custom_model.parameters(), lr = 1e-3)
custom_model_train_loss, custom_model_test_loss = [], []
custom_model_train_accs, custom_model_test_accs = [], []


torch.manual_seed(64)
torch.cuda.manual_seed(64)
start_time = timer()
for epoch in tqdm(range(epoches)):

  train_loss, train_acc = train_loop(model = custom_model, dataloader = train_dataloader,
                                    loss_fn = loss_fn, optimizer = optimizer,
                                    accuracy_fn = accuracy_fn, device = device)
  
  test_loss, test_acc = test_loop(model = custom_model, dataloader = test_dataloader,
                                  loss_fn = loss_fn, accuracy_fn = accuracy_fn,
                                  device = device)
  
  custom_model_train_loss.append(train_loss.item())
  custom_model_test_loss.append(test_loss.item())
  custom_model_train_accs.append(train_acc.item())
  custom_model_test_accs.append(test_acc.item())

  print(f"Epoch: {epoch+1}  Train Loss: {train_loss:.4f} | Test Loss: {test_loss:.4f} | Train Accuray: {train_acc:.4f} | Test Accuracy: {test_acc:.4f}")

end_time = timer()
plot_graph(custom_model_train_loss, custom_model_test_loss, custom_model_train_accs, custom_model_test_accs)
print(f"train ing time: {end_time - start_time}")

# save_model('cifair10_custom_model.pth', custom_model)
# print("Model saved")

# optimizer
optimizer = torch.optim.Adam(params = custom_model.parameters(), lr = 1e-3)
with torch.profiler.profile(
        schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=2),
        on_trace_ready=torch.profiler.tensorboard_trace_handler('./logs/custom_model'),
        record_shapes=True,
        profile_memory=True,
        with_stack=True
) as prof:
    for step, batch_data in enumerate(train_dataloader):
        if step >= (1 + 1 + 3) * 2:
            break
        train(batch_data, custom_model, loss_fn, optimizer)
        prof.step()

# Commented out IPython magic to ensure Python compatibility.
try:
#   %load_ext tensorboard
except:
  print("reload tensor board")
else:
#   %reload_ext tensorboard
#%tensorboard --logdir log
# %tensorboard --logdir ./logs

"""### VGG16"""

vgg16 = models.vgg16(pretrained = False, progress = False).to(device)
vgg16.classifier[6] = nn.Linear(4096,num_class).to(device)
vgg16

"""#### train model"""

epoches = 5

vgg16 = models.vgg16(pretrained = False, progress = False).to(device)
vgg16.classifier[6] = nn.Linear(4096,num_class).to(device)

# optimizer
optimizer = torch.optim.Adam(params = vgg16.parameters(), lr = 1e-3)
vgg16_train_loss, vgg16_test_loss = [], []
vgg16_train_accs, vgg16_test_accs = [], []


torch.manual_seed(64)
torch.cuda.manual_seed(64)
start_time = timer()
for epoch in tqdm(range(epoches)):

  train_loss, train_acc = train_loop(model = vgg16, dataloader = train_dataloader,
                                    loss_fn = loss_fn, optimizer = optimizer,
                                    accuracy_fn = accuracy_fn, device = device)
  
  test_loss, test_acc = test_loop(model = vgg16, dataloader = test_dataloader,
                                  loss_fn = loss_fn, accuracy_fn = accuracy_fn,
                                  device = device)
  
  vgg16_train_loss.append(train_loss.item())
  vgg16_test_loss.append(test_loss.item())
  vgg16_train_accs.append(train_acc.item())
  vgg16_test_accs.append(test_acc.item())

  print(f"Epoch: {epoch+1}  Train Loss: {train_loss:.4f} | Test Loss: {test_loss:.4f} | Train Accuray: {train_acc:.4f} | Test Accuracy: {test_acc:.4f}")

end_time = timer()
plot_graph(vgg16_train_loss, vgg16_test_loss, vgg16_train_accs, vgg16_test_accs)
print(f"train ing time: {end_time - start_time}")

# save_model('cifair10_custom_model.pth', custom_model)
# print("Model saved")

# optimizer
optimizer = torch.optim.Adam(params = vgg16.parameters(), lr = 1e-3)
with torch.profiler.profile(
        schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=2),
        on_trace_ready=torch.profiler.tensorboard_trace_handler('./logs/vgg16'),
        record_shapes=True,
        profile_memory=True,
        with_stack=True
) as prof:
    for step, batch_data in enumerate(train_dataloader):
        if step >= (1 + 1 + 3) * 2:
            break
        train(batch_data, vgg16, loss_fn, optimizer)
        prof.step()

"""# Increase the Model performance and reduce the training time

## reduce CNN channel
"""

class CustomCNN(nn.Module):
  def __init__(self, in_channel: int, out_channel: int, out_feature: int):
    super().__init__()
    self.conv_block = nn.Sequential(
                                        nn.Conv2d(in_channels = in_channel, out_channels = out_channel, kernel_size = 3, stride = 1),
                                        nn.ReLU(),
                                        nn.Conv2d(in_channels = out_channel, out_channels = out_channel, kernel_size = 3, stride = 1),
                                        nn.ReLU(),
                                        nn.MaxPool2d(kernel_size = 2, stride = 2),
                                        nn.Conv2d(in_channels = out_channel, out_channels = out_channel, kernel_size = 3, stride = 1),
                                        nn.ReLU(),
                                        nn.MaxPool2d(kernel_size = 2, stride = 2),
                                        nn.Conv2d(in_channels = out_channel, out_channels = out_channel, kernel_size = 3, stride = 1),
                                        nn.ReLU(),
                                        nn.MaxPool2d(kernel_size = 2, stride = 2)
                                  )
    self.linear_block = nn.Sequential(
                                        nn.Flatten(),
                                        nn.Linear(in_features = 10*2*2, out_features = 1024),
                                        nn.ReLU(),
                                        nn.Linear(in_features = 1024, out_features = out_feature)                                        
                                    )
    
  def forward(self, x):
    conv_x = self.conv_block(x)
    # print(f"conv_x shape: {conv_x.shape}")
    out = self.linear_block(conv_x)

    return out

custom_model_u1 = CustomCNN(3, 10, num_class).to(device)
custom_model_u1

# init. epochs
epoches = 11

custom_model_u1 = CustomCNN(3, 10, num_class).to(device)

# optimizer
optimizer = torch.optim.Adam(params = custom_model_u1.parameters(), lr = 1e-3)
custom_model_u1_train_loss, custom_model_u1_test_loss = [], []
custom_model_u1_train_accs, custom_model_u1_test_accs = [], []


torch.manual_seed(64)
torch.cuda.manual_seed(64)
start_time = timer()
for epoch in tqdm(range(epoches)):

  train_loss, train_acc = train_loop(model = custom_model_u1, dataloader = train_dataloader,
                                    loss_fn = loss_fn, optimizer = optimizer,
                                    accuracy_fn = accuracy_fn, device = device)
  
  test_loss, test_acc = test_loop(model = custom_model_u1, dataloader = test_dataloader,
                                  loss_fn = loss_fn, accuracy_fn = accuracy_fn,
                                  device = device)
  
  custom_model_u1_train_loss.append(train_loss.item())
  custom_model_u1_test_loss.append(test_loss.item())
  custom_model_u1_train_accs.append(train_acc.item())
  custom_model_u1_test_accs.append(test_acc.item())

  print(f"Epoch: {epoch+1}  Train Loss: {train_loss:.4f} | Test Loss: {test_loss:.4f} | Train Accuray: {train_acc:.4f} | Test Accuracy: {test_acc:.4f}")

end_time = timer()
plot_graph(custom_model_u1_train_loss, custom_model_u1_test_loss, custom_model_u1_train_accs, custom_model_u1_test_accs)
print(f"train ing time: {end_time - start_time}")

# save_model('cifair10_custom_model.pth', custom_model)
# print("Model saved")

# optimizer
optimizer = torch.optim.Adam(params = custom_model_u1.parameters(), lr = 1e-3)
with torch.profiler.profile(
        schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=2),
        on_trace_ready=torch.profiler.tensorboard_trace_handler('./logs/custom_model_u1'),
        record_shapes=True,
        profile_memory=True,
        with_stack=True
) as prof:
    for step, batch_data in enumerate(train_dataloader):
        if step >= (1 + 1 + 3) * 2:
            break
        train(batch_data, custom_model_u1, loss_fn, optimizer)
        prof.step()

# try:
#   %load_ext tensorboard
# except:
#   print("reload tensor board")
# else:
#   %reload_ext tensorboard
# #%tensorboard --logdir log
# %tensorboard --logdir ./logs

"""## Add pooling layer"""

class CustomCNN(nn.Module):
  def __init__(self, in_channel: int, out_channel: int, out_feature: int):
    super().__init__()
    self.conv_block = nn.Sequential(
                                        nn.Conv2d(in_channels = in_channel, out_channels = out_channel, kernel_size = 5, stride = 2),
                                        nn.ReLU(),
                                        # nn.MaxPool2d(kernel_size = 2, stride = 2),
                                        # nn.Conv2d(in_channels = out_channel, out_channels = out_channel, kernel_size = 4, stride = 2),
                                        # nn.ReLU(),
                                        # nn.MaxPool2d(kernel_size = 2, stride = 2),
                                        # nn.Conv2d(in_channels = out_channel, out_channels = out_channel, kernel_size = 5, stride = 2),
                                        # nn.ReLU(),
                                        nn.MaxPool2d(kernel_size = 2, stride = 2),
                                        nn.Conv2d(in_channels = out_channel, out_channels = out_channel, kernel_size = 3, stride = 1),
                                        nn.ReLU(),
                                        nn.MaxPool2d(kernel_size = 5, stride = 2)
                                  )
    self.linear_block = nn.Sequential(
                                        nn.Flatten(),
                                        nn.Linear(in_features = 20*1*1, out_features = 1024),
                                        nn.ReLU(),
                                        nn.Linear(in_features = 1024, out_features = out_feature)                                        
                                    )
    
  def forward(self, x):
    conv_x = self.conv_block(x)
    # print(f"conv_x shape: {conv_x.shape}")
    out = self.linear_block(conv_x)

    return out

custom_model_u2 = CustomCNN(3, 20, num_class).to(device)
custom_model_u2

# init. epochs
epoches = 11

custom_model_u2 = CustomCNN(3, 20, num_class).to(device)

# optimizer
optimizer = torch.optim.Adam(params = custom_model_u2.parameters(), lr = 1e-3)
custom_model_u2_train_loss, custom_model_u2_test_loss = [], []
custom_model_u2_train_accs, custom_model_u2_test_accs = [], []


torch.manual_seed(64)
torch.cuda.manual_seed(64)
start_time = timer()
for epoch in tqdm(range(epoches)):

  train_loss, train_acc = train_loop(model = custom_model_u2, dataloader = train_dataloader,
                                    loss_fn = loss_fn, optimizer = optimizer,
                                    accuracy_fn = accuracy_fn, device = device)
  
  test_loss, test_acc = test_loop(model = custom_model_u2, dataloader = test_dataloader,
                                  loss_fn = loss_fn, accuracy_fn = accuracy_fn,
                                  device = device)
  
  custom_model_u2_train_loss.append(train_loss.item())
  custom_model_u2_test_loss.append(test_loss.item())
  custom_model_u2_train_accs.append(train_acc.item())
  custom_model_u2_test_accs.append(test_acc.item())

  print(f"Epoch: {epoch+1}  Train Loss: {train_loss:.4f} | Test Loss: {test_loss:.4f} | Train Accuray: {train_acc:.4f} | Test Accuracy: {test_acc:.4f}")

end_time = timer()
plot_graph(custom_model_u2_train_loss, custom_model_u2_test_loss, custom_model_u2_train_accs, custom_model_u2_test_accs)
print(f"train ing time: {end_time - start_time}")

# save_model('cifair10_custom_model.pth', custom_model)
# print("Model saved")

# optimizer
optimizer = torch.optim.Adam(params = custom_model_u2.parameters(), lr = 1e-3)
with torch.profiler.profile(
        schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=2),
        on_trace_ready=torch.profiler.tensorboard_trace_handler('./logs/custom_model_u2'),
        record_shapes=True,
        profile_memory=True,
        with_stack=True
) as prof:
    for step, batch_data in enumerate(train_dataloader):
        if step >= (1 + 1 + 3) * 2:
            break
        train(batch_data, custom_model_u2, loss_fn, optimizer)
        prof.step()

# Commented out IPython magic to ensure Python compatibility.
try:
#   %load_ext tensorboard
except:
  print("reload tensor board")
else:
#   %reload_ext tensorboard
#%tensorboard --logdir log
# %tensorboard --logdir ./logs

# https://towardsdatascience.com/how-to-reduce-training-parameters-in-cnns-while-keeping-accuracy-99-a213034a9777
# https://medium.com/@dipti.rohan.pawar/improving-performance-of-convolutional-neural-network-2ecfe0207de7
# https://www.analyticsvidhya.com/blog/2019/11/4-tricks-improve-deep-learning-model-performance/