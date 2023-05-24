import torch
from torch import nn

import torchvision
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import optuna
from optuna.trial import TrialState

from tqdm.auto import tqdm
from timeit import default_timer as timer

from torchmetrics.classification import MulticlassAccuracy

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f'device: {device}')


transform = transforms.Compose([
                                transforms.Resize((64, 64)),
                                transforms.ToTensor(),
                              ])

# load data
train_dataset = datasets.FashionMNIST(root = './data', train = True, transform = transform, download = True)
test_dataset = datasets.FashionMNIST(root = './data', train = False, transform = transform, download = True)


# make datasets
BATCH_SIZE = 32
train_dataloader = DataLoader(dataset = train_dataset, batch_size = BATCH_SIZE, shuffle = True)
test_dataloader = DataLoader(dataset = test_dataset, batch_size = BATCH_SIZE, shuffle = False)


print(f'Total {len(train_dataloader)} of train each of {BATCH_SIZE} batches.')
print(f'Total {len(test_dataloader)} of test each of {BATCH_SIZE} batches.')


# class name
class_names = train_dataset.classes
# class_names

index_cls = train_dataset.class_to_idx
# index_cls


# Train and Test Loop
# train
def train_loop(model: torch.nn.Module, dataloader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module, optimizer: torch.optim.Optimizer,
               accuracy_fn, device: torch.device = device):
  
  train_loss, train_acc = 0, 0

  model.train()

  for batch, (x_train, y_train) in enumerate(dataloader):

    if device == 'cuda':
      x_train, y_train = x_train.to(device), y_train.to(device)
    

    # 1. Forward step
    # print("image: ",type(x_train))
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


# loss and accuracy function
loss_fn = nn.CrossEntropyLoss()

accuracy_fn = MulticlassAccuracy(num_classes = len(class_names)).to(device)

# build model 
class CustomCNN(nn.Module):
    
    def __init__(self, trial, num_conv_layers, num_filters, num_neurons, drop_conv2, drop_fc1):
        
        super().__init__()                                                     # Initialize parent class
        in_size = 32                                                                    # Input image size (28 pixels)
        kernel_size = 3                                                                 # Convolution filter size


        self.conv2_drop = nn.Dropout2d(p=drop_conv2)                                    # Dropout for conv2
        self.out_feature = num_filters * 64 * 64 # out_size         # Size of flattened features
        
        layers = []

        for i in range(num_conv_layers):
          if i == 0:
            layers.append(nn.Conv2d(in_channels=1, out_channels=num_filters, kernel_size=(3, 3), padding = 1, stride = 1))
            layers.append(nn.ReLU()) 

          else:
            if i%2 == 0:
              layers.append(nn.Conv2d(in_channels=num_filters, out_channels=num_filters, kernel_size=(3, 3), padding = 1, stride = 1))
              layers.append(nn.ReLU())
              layers.append(nn.MaxPool2d(kernel_size = (3,3), stride = 1, padding = 1))

            else:
              layers.append(nn.Conv2d(in_channels=num_filters, out_channels=num_filters, kernel_size=(3, 3), padding = 1, stride = 1))
              layers.append(nn.ReLU())

        self.cnn_layer = nn.Sequential(*layers)
        
        # print(self.cnn_layer)
        self.fc1 = nn.Sequential(
                                  nn.Flatten(),
                                  nn.Linear(self.out_feature, num_neurons),                            # Fully Connected layer 1
                                  nn.ReLU(),
                                  nn.Linear(num_neurons, 10)                                           # Fully Connected layer 2
                                )

    def forward(self, x):
        
        x = self.cnn_layer(x)
        # print("x: ", x.shape, x.shape[-1], )
        # self.out_feature = x.shape[-1] * x.shape[-2] * x.shape[-3]

        x = self.fc1(x)  # Flatten tensor

        return x 


def objective(trial, n_trials=100):
    

    # Define range of values to be tested for the hyperparameters
    num_conv_layers = trial.suggest_int("num_conv_layers", 3, 6)  # Number of convolutional layers
 

    # Generate the model
    model = CustomCNN(trial, num_conv_layers, num_filters = 10, num_neurons = 512, drop_conv2 = 0.2,  drop_fc1 = 0.2).to(device)


    lr = trial.suggest_float("lr", 1e-4, 1e-1, log=True)                                 # Learning rates
    n_epochs = trial.suggest_int('n_epochs', 10, 20)
    optimizer = torch.optim.Adam(model.parameters(), lr = lr)

    # Training of the model
    for epoch in range(n_epochs):
        train_loss, train_acc = train_loop(model, train_dataloader, loss_fn, optimizer,
                                            accuracy_fn, device = device)  # Train the model
        test_loss, test_acc = test_loop(model, test_dataloader, loss_fn, accuracy_fn, device = device)   # Evaluate the model

        # For pruning (stops trial early if not promising)
        trial.report(test_acc, epoch)
        # Handle pruning based on the intermediate value.
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return test_acc


# Create an Optuna study to maximize test accuracy
study = optuna.create_study(direction="maximize")
print(study.optimize(objective, n_trials=100))


# Find number of pruned and completed trials
pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

# Display the study statistics
print("\nStudy statistics: ")
print("  Number of finished trials: ", len(study.trials))
print("  Number of pruned trials: ", len(pruned_trials))
print("  Number of complete trials: ", len(complete_trials))

trial = study.best_trial
print("Best trial:")
print("  Value: ", trial.value)
print("  Params: ")
for key, value in trial.params.items():
    print("    {}: {}".format(key, value))