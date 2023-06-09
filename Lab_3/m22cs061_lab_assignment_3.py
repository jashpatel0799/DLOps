# -*- coding: utf-8 -*-
"""M22CS061_Lab_Assignment_3.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1W4w7Q2f4KDlut7DJ8EYYEL4KD6UQQaho

## Import Libraries
"""

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

!pip install -q torchmetrics
from torchmetrics.classification import MulticlassAccuracy

"""## device agnostic code"""

# device agnostic code
device = 'cuda' if torch.cuda.is_available() else 'cpu'
device

"""## Load pre train model on imagenet"""

model_18 = models.resnet18(weights = models.ResNet18_Weights.IMAGENET1K_V1)
model_best_transform = models.ResNet18_Weights.IMAGENET1K_V1.transforms()

model_18, model_best_transform

feature_number = model_18.fc.in_features
model_18.fc = nn.Linear(feature_number, 10).to(device)

"""## Load Dataset"""

train_data = datasets.STL10(root = 'data', split = "train", transform = model_best_transform, download = True)
test_data = datasets.STL10(root = 'data', split = "test", transform = model_best_transform, download = True)

print(len(train_data)), print(len(test_data))

"""## Make dataloader"""

# setup batchsize hyperparameter
BATCH_SIZE = 32

# turn dataset into batches
train_dataloader = DataLoader(train_data, batch_size = BATCH_SIZE, shuffle = True, drop_last = True)

test_dataloader = DataLoader(test_data, batch_size = BATCH_SIZE, shuffle = False, drop_last = True)

print(f"DataLoader: {train_dataloader}, {test_dataloader}")
print(f"Length of train dataloader: {len(train_dataloader)} batches of {BATCH_SIZE}")
print(f"Length of test dataloader: {len(test_dataloader)} batches of {BATCH_SIZE}")

"""## visulize data"""

class_names = train_data.classes
class_names

image, label = train_data[0]
print(f"Image: {image.shape}")
plt.imshow(image.permute(1,2,0))
plt.title(class_names[label])
# image

"""## Train and Test Loop"""

# train
def train_loop(model: torch.nn.Module, dataloader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module, optimizer: torch.optim.Optimizer,
               accuracy_fn, device: torch.device = device):
  
  train_loss, train_acc = 0, 0
  for batch, (x_train, y_train) in enumerate(dataloader):

    if device == 'cuda':
      x_train, y_train = x_train.to(device), y_train.to(device)

    model.train()

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

"""## Plot graph of Loss and Accuray"""

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

"""## Loss and Accuracy function"""

loss_fn = nn.CrossEntropyLoss()

accuracy_fn = MulticlassAccuracy(num_classes = len(class_names)).to(device)

"""# Train Model on RMSProp"""

# Hyperparms
lr_rate = [0.1,0.01] # learning rate
alpha = [0.85,0.95] # smoothing constant
momentum = [0.88,0.99] # momentum factor
eps = [1e-5,1e-7] # term added to the denominator to improve numerical stability

hyper_params = [(lr,a,m,e) for lr in lr_rate for a in alpha for m in momentum for e in eps]
# print(parms_combs)

cur_iter,total_iter = 1, len(lr_rate)*len(alpha)*len(momentum)*len(eps)

# init. epochs
epoches = 5

for h_p in hyper_params:
  print()
  print(f"current exp | total: {cur_iter} | {total_iter}")
  print(f"Training with --- lr rate: {h_p[0]}, alpha: {h_p[1]}, momentum: {h_p[2]}, eps: {h_p[3]}")
  model_18 = models.resnet18(weights = models.ResNet18_Weights.IMAGENET1K_V1).to(device)
  feature_number = model_18.fc.in_features
  model_18.fc = nn.Linear(feature_number, len(class_names)).to(device)

  optimizer = torch.optim.RMSprop(model_18.parameters(), h_p[0], h_p[1], h_p[2], h_p[3])
  model18_train_loss, model18_test_loss = [], []
  model18_train_accs, model18_test_accs = [], []

  torch.manual_seed(64)
  torch.cuda.manual_seed(64)
  for epoch in tqdm(range(epoches)):

    train_loss, train_acc = train_loop(model = model_18, dataloader = train_dataloader,
                                      loss_fn = loss_fn, optimizer = optimizer,
                                      accuracy_fn = accuracy_fn, device = device)
    
    test_loss, test_acc = test_loop(model = model_18, dataloader = test_dataloader,
                                    loss_fn = loss_fn, accuracy_fn = accuracy_fn,
                                    device = device)
    
    model18_train_loss.append(train_loss.item())
    model18_test_loss.append(test_loss.item())
    model18_train_accs.append(train_acc.item())
    model18_test_accs.append(test_acc.item())

    print(f"Epoch: {epoch+1}  Train Loss: {train_loss:.4f} | Test Loss: {test_loss:.4f} | Train Accuray: {train_acc:.4f} | Test Accuracy: {test_acc:.4f}")
  cur_iter += 1
  plot_graph(model18_train_loss, model18_test_loss, model18_train_accs, model18_test_accs)

"""# Train Model on Adam"""

# Hyperparameter
lr_rate = [0.001,0.0001] # learning rate
betas=[(0.85, 0.8585),(0.95, 0.9595)] # coefficients used for computing running averages of gradient and its square
eps = [1e-5,1e-7] # term added to the denominator to improve numerical stability
weight_decay = [0.001,0.0005] # weight decay (L2 penalty)

hyper_params = [(lr,b,e,wd) for lr in lr_rate for b in betas for e in eps for wd in weight_decay]
# print(parms_combs)

cur_iter,total_iter = 1, len(lr_rate)*len(betas)*len(eps)*len(weight_decay)

# init. epochs
epoches = 5

for h_p in hyper_params:
  print()
  print(f"current exp | total: {cur_iter} | {total_iter}")
  print(f"Training with --- lr rate: {h_p[0]}, betas: {h_p[1]}, eps: {h_p[2]}, weight decay: {h_p[3]}")
  model_18 = models.resnet18(weights = models.ResNet18_Weights.IMAGENET1K_V1).to(device)
  feature_number = model_18.fc.in_features
  model_18.fc = nn.Linear(feature_number, len(class_names)).to(device)

  optimizer = torch.optim.Adam(params=model_18.parameters(), lr=h_p[0], betas=h_p[1], eps=h_p[2],weight_decay=h_p[3])
  model18_train_loss, model18_test_loss = [], []
  model18_train_accs, model18_test_accs = [], []

  torch.manual_seed(64)
  torch.cuda.manual_seed(64)
  for epoch in tqdm(range(epoches)):

    train_loss, train_acc = train_loop(model = model_18, dataloader = train_dataloader,
                                      loss_fn = loss_fn, optimizer = optimizer,
                                      accuracy_fn = accuracy_fn, device = device)
    
    test_loss, test_acc = test_loop(model = model_18, dataloader = test_dataloader,
                                    loss_fn = loss_fn, accuracy_fn = accuracy_fn,
                                    device = device)
    
    model18_train_loss.append(train_loss.item())
    model18_test_loss.append(test_loss.item())
    model18_train_accs.append(train_acc.item())
    model18_test_accs.append(test_acc.item())

    print(f"Epoch: {epoch+1}  Train Loss: {train_loss:.4f} | Test Loss: {test_loss:.4f} | Train Accuray: {train_acc:.4f} | Test Accuracy: {test_acc:.4f}")
  cur_iter += 1
  plot_graph(model18_train_loss, model18_test_loss, model18_train_accs, model18_test_accs)

"""# Train model on Adadelta"""

# Hyperparameters
lr_rate = [0.01, 0.001] # learning rate
rho = [0.75,0.85] # coefficient used for computing a running average of squared gradients
eps = [1e-5,1e-7] # term added to the denominator to improve numerical stability
weight_decay = [0.001,0.0001] # weight decay (L2 penalty)

hyper_params = [(lr,r,e,wd) for lr in lr_rate for r in rho for e in eps for wd in weight_decay]
# print(parms_combs)

cur_iter,total_iter = 1, len(lr_rate)*len(rho)*len(eps)*len(weight_decay)

# init. epochs
epoches = 5

for h_p in hyper_params:
  print()
  print(f"current exp | total: {cur_iter} | {total_iter}")
  print(f"Training with --- lr rate: {h_p[0]}, rho: {h_p[1]}, eps: {h_p[2]}, weight decay: {h_p[3]}")
  model_18 = models.resnet18(weights = models.ResNet18_Weights.IMAGENET1K_V1).to(device)
  feature_number = model_18.fc.in_features
  model_18.fc = nn.Linear(feature_number, len(class_names)).to(device)

  optimizer = torch.optim.Adadelta(model_18.parameters(), h_p[0], h_p[1], h_p[2], h_p[3])
  model18_train_loss, model18_test_loss = [], []
  model18_train_accs, model18_test_accs = [], []

  torch.manual_seed(64)
  torch.cuda.manual_seed(64)
  for epoch in tqdm(range(epoches)):

    train_loss, train_acc = train_loop(model = model_18, dataloader = train_dataloader,
                                      loss_fn = loss_fn, optimizer = optimizer,
                                      accuracy_fn = accuracy_fn, device = device)
    
    test_loss, test_acc = test_loop(model = model_18, dataloader = test_dataloader,
                                    loss_fn = loss_fn, accuracy_fn = accuracy_fn,
                                    device = device)
    
    model18_train_loss.append(train_loss.item())
    model18_test_loss.append(test_loss.item())
    model18_train_accs.append(train_acc.item())
    model18_test_accs.append(test_acc.item())

    print(f"Epoch: {epoch+1}  Train Loss: {train_loss:.4f} | Test Loss: {test_loss:.4f} | Train Accuray: {train_acc:.4f} | Test Accuracy: {test_acc:.4f}")
  cur_iter += 1
  plot_graph(model18_train_loss, model18_test_loss, model18_train_accs, model18_test_accs)