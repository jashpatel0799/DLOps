import torch
from torch import nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder


from torchmetrics.classification import MulticlassAccuracy



# Plot graph of Loss and Accuray
def plot_graph(train_loss, test_loss, train_acc, test_acc):
  plt.figure(figsize = (20, 8))
  plt.subplot(1, 2, 1)
  plt.plot(range(len(train_loss)), train_loss, label = "Train Loss")
  plt.plot(range(len(test_loss)), test_loss, label = "Test Loss")
  plt.legend()
  plt.xlabel("Epoches")
  plt.ylabel("Loss")
  # plt.show()

  plt.subplot(1, 2, 2)
  plt.plot(range(len(train_acc)), train_acc, label = "Train Accuracy")
  plt.plot(range(len(test_acc)), test_acc, label = "Test Accuracy")
  plt.legend()
  plt.xlabel("Epoches")
  plt.ylabel("Accuracy")
  plt.show()


# Build model
class IrisNN(nn.Module):
  def __init__(self, input_features, output_features):
    super().__init__()

    self.network = nn.Sequential(
        nn.Linear(in_features = input_features, out_features = 4),
        nn.ReLU(),
        nn.Linear(in_features = 4, out_features = 5),
        nn.ReLU(),
        nn.Linear(in_features = 5, out_features = output_features)
    )

  def forward(self, x):
    return self.network(x)

# model_iris = IrisNN(input_features = 4, output_features = 3).to(device)
# model_iris


# Early stopping
class EarlyStopping:
    def __init__(self, tolerance=10, min_delta=5):

        self.tolerance = tolerance
        self.min_delta = min_delta
        self.counter = 0
        self.early_stop = False

    def __call__(self, train_loss, validation_loss):
        if (validation_loss - train_loss) > self.min_delta:
            self.counter +=1
            if self.counter >= self.tolerance:  
                self.early_stop = True


# device agnostic code
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# print(device)

# get dataset
info = load_iris()
x = info.data
y = info.target

x_torch = torch.from_numpy(x).type(torch.float)
y_torch = torch.from_numpy(y).type(torch.long)
x_train, x_test, y_train, y_test = train_test_split(x_torch, y_torch, test_size = 0.2, random_state = 64)
x_train, y_train = x_train.to(device), y_train.to(device)
x_test, y_test = x_test.to(device), y_test.to(device)


# loss and accuracy function
loss_fn = nn.CrossEntropyLoss()

accuracy_fn = MulticlassAccuracy(num_classes = 3).to(device)

early_stopping = EarlyStopping(tolerance=10, min_delta=5)

model_iris = IrisNN(input_features = 4, output_features = 3).to(device)

optimizer = torch.optim.SGD(params = model_iris.parameters(), lr = 0.01)

# training loop
torch.manual_seed(42)
torch.cuda.manual_seed(42)
epoches = 2000

train_loss_list = []
test_loss_list = []

train_acc_list = []
test_acc_list = []

for epoch in range(epoches):
  # train
  model_iris.train()
  # 1.Forward
  y_logits = model_iris(x_train)
  y_pred = torch.argmax(y_logits, dim = 1)

  # 2.Loss
  # print(y_logits.shape)
  # print(type(y_logits))
  # print(y_train.shape)
  # print(type(y_train))
  # print(y_pred.shape)
  # print(y_pred)
  loss = loss_fn(y_logits, y_train)
  acc = accuracy_fn(y_pred, y_train)

  # 3. Grad zero
  optimizer.zero_grad()

  # 4. Backward
  loss.backward()

  # 5. grad step
  optimizer.step()

  # test
  model_iris.eval()
  with torch.inference_mode():
    # 1. Forward
    test_logits = model_iris(x_test).squeeze()
    test_pred = torch.argmax(test_logits, dim = 1)

    # 2. loss
    test_loss = loss_fn(test_logits, y_test)
    # print(test_pred,y_test)
    test_acc = accuracy_fn(test_pred, y_test)

  if epoch % 200 == 0:
    print(f"Epoch: {epoch} | Loss: {loss:.4f} | Test Loss: {test_loss:.4f} | acc: {acc:.4f} | test_acc: {test_acc:.4f}")
    train_loss_list.append(loss.item())
    test_loss_list.append(test_loss.item())
    train_acc_list.append(acc.item())
    test_acc_list.append(test_acc.item())

  # early stopping
  early_stopping(loss, test_loss)
  if early_stopping.early_stop:
    print("We are at epoch:", epoch)
    break

plot_graph(train_loss_list, test_loss_list, train_acc_list, test_acc_list)