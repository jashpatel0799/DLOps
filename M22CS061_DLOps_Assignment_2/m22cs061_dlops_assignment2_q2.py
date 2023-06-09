# -*- coding: utf-8 -*-
"""M22CS061_DLOps_Assignment2_Q2.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1VYKaL7krmT6K_PdnZpyWlPzrLPvCjeO2
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader, Dataset
from tqdm.auto import tqdm
import gc

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from pathlib import Path
from google.colab import drive
drive.mount('/content/drive/')

"""## Write device agnostic code"""

device = 'cuda' if torch.cuda.is_available() else 'cpu'
device

data = 'drive/MyDrive/Course/Sem2/DLOps/data/household_power_consumption.txt'
info = df = pd.read_csv(data, sep=';', na_values=['?'],)
info.head()

len(info.index)

"""## PreProcess the data"""

info.drop(info.columns[[0,1]], axis=1, inplace=True)
# info.drop(info.columns[[0]], axis=1, inplace=True)

info.head()

col_num = len(info.columns)
def check_null(col_num, dataframe):
  for c in range(col_num):
    print(f"Null value in {dataframe.columns[c]}: {dataframe.iloc[:,c].isnull().sum()}")

# def check_zeros(col_num, dataframe):
#   for c in range(col_num):
#     print(f"Zero value in {dataframe.columns[c]}: {(dataframe.iloc[:,c] == 0).sum()}")



check_null(col_num, info)
# print()
# print("----------------------------------------------------------------------------")
# print()
# check_zeros(col_num, info)

"""### Fill Null value with Meadian"""

# df['Sub_metering_3'] = df['Sub_metering_3'].fillna(df['Sub_metering_3'].median())
info = info.dropna()

len(info.index)

"""### Check for null values"""

check_null(col_num, info)

"""# Split data into train test 80:20"""

info = info.astype(float)

info_numpy = info.to_numpy()
x = info_numpy[:,1:]
y = info_numpy[:,0:1]

print(x)
print()
print(y)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=64)

"""### Now Apply standard scaller on train and test data"""



data_scaler = StandardScaler()
x_train = data_scaler.fit_transform(x_train)
x_test = data_scaler.transform(x_test)


# info_scaler = pd.DataFrame(info_scaler, columns=['Global_active_power','Global_reactive_power','Voltage','Global_intensity', 'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3'])
print(x_train)
print()
print(x_test)

pred_scaler = StandardScaler()
pred_scaler.fit(y_train)
y_train = pred_scaler.transform(y_train)
y_test = pred_scaler.transform(y_test)


print(y_train)
print()
print(y_test)

"""## Make DataLoader"""

class Makedataset(Dataset):
    def __init__(self, x, y, sequence_length=5):
        
        self.sequence_length = sequence_length
        self.X = x
        self.y = y

    def __len__(self):
        
        return self.X.shape[0]

    def __getitem__(self, i): 
        
        if i >= self.sequence_length - 1:
            i_start = i - self.sequence_length + 1
            x = self.X[i_start:(i + 1), :]
        else:
            padding = self.X[0].repeat(self.sequence_length - i - 1, 1)
            x = self.X[0:(i + 1), :]
            x = torch.cat((padding, x), 0)

        return x, self.y[i].item()

def numpy_to_dataLoader(x, y, batch_size, shuffle, sequence_num=5):
  # transform to torch tensor
  tensor_x = torch.Tensor(x) 
  tensor_y = torch.Tensor(y)

  # create your datset
  my_dataset = Makedataset(tensor_x,tensor_y) 

  # create your dataloader
  my_dataloader = DataLoader(my_dataset, batch_size = batch_size, shuffle = shuffle) 

  return my_dataloader

BATCH_SIZE = 64

train_dataloader = numpy_to_dataLoader(x_train, y_train, batch_size = BATCH_SIZE, shuffle = True)
test_dataloader = numpy_to_dataLoader(x_test, y_test, batch_size = BATCH_SIZE, shuffle = False)

for i, (x, y) in enumerate(train_dataloader):
  print(len(x))
  break

"""## Train and Test Loop"""

def train_loop(model: torch.nn.Module, dataloader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module, optimizer: torch.optim.Optimizer,
               device: torch.device = device):
  train_loss = 0
  model.train()
  for batch, (x_train, y_train) in enumerate(dataloader):

    if device == 'cuda':
      x_train, y_train = x_train.to(device), y_train.to(device)

    y_train = y_train.type(torch.float32)

    # model.train()
    optimizer.zero_grad()

    pred = model(x_train)

    loss = loss_fn(pred, y_train)


    loss.backward()

    optimizer.step()

    train_loss += loss

  train_loss /= len(dataloader)

  return train_loss
def test_loop(model: torch.nn.Module, dataloader: torch.utils.data.DataLoader,
              loss_fn: torch.nn.Module, device: torch.device = device):
  
  test_loss = 0

  model.eval()
  with torch.inference_mode():
    for x_test, y_test in dataloader:

      if device == 'cuda':
        x_test, y_test = x_test.to(device), y_test.to(device)

      y_test = y_test.type(torch.float32)

      test_pred = model(x_test)

      test_loss += loss_fn(test_pred, y_test)

    test_loss /= len(dataloader)

  return test_loss

def eval_loop(model: torch.nn.Module, dataloader: torch.utils.data.DataLoader,
              loss_fn: torch.nn.Module, device: torch.device = device):
  
  
  test_val = []

  model.eval()
  with torch.inference_mode():
    for x_test, y_test in dataloader:

      if device == 'cuda':
        x_test, y_test = x_test.to(device), y_test.to(device)

      y_test = y_test.type(torch.float32)

      # 1. Forward
      test_pred = model(x_test)

      for i in test_pred:
        test_val.append(i.item())
    # test_acc /= len(dataloader)


  return test_val

"""## Plot functon for Loss/Accuray vs Epoch"""

def plot_graph(train_losses, test_losses):
  plt.figure(figsize = (10, 8))
  # plt.subplot(1, 2, 1)
  plt.plot(range(len(train_losses)), train_losses, label = "Train Loss")
  plt.plot(range(len(test_losses)), test_losses, label = "Test Loss")
  plt.legend()
  plt.xlabel("Epoches")
  plt.ylabel("Loss")
  # plt.show()

  # plt.subplot(1, 2, 2)
  # plt.plot(range(len(train_accs)), train_accs, label = "Train Accuracy")
  # plt.plot(range(len(test_accs)), test_accs, label = "Test Accuracy")
  # plt.legend()
  # plt.xlabel("Epoches")
  # plt.ylabel("Accuracy")
  # plt.show()

"""## Loss function"""

loss_fn = nn.MSELoss()

"""## Build Model"""

class LSTM(nn.Module):
    def __init__(self, num_sensors, hidden_units):
        super().__init__()
        self.num_sensors = num_sensors  # this is the number of features
        self.hidden_units = hidden_units
        self.num_layers = 1

        self.lstm = nn.LSTM(input_size=num_sensors, hidden_size=hidden_units,
                            batch_first=True, num_layers=self.num_layers)

        self.linear = nn.Linear(in_features=self.hidden_units, out_features=1)

    def forward(self, x):
        batch_size = x.shape[0]
        # h0 = torch.zeros(self.num_layers, batch_size, self.hidden_units).squeeze().requires_grad_()
        # c0 = torch.zeros(self.num_layers, batch_size, self.hidden_units).squeeze().requires_grad_()

        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_units).requires_grad_().to(device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_units).requires_grad_().to(device)

        # print(f'h0: {h0.shape}')
        
        _, (hn, cn) = self.lstm(x, (h0, c0))
        # print(f'hn: {hn.shape}')
        # print(f'cn: {cn.shape}')
        out = self.linear(hn[0]).flatten()  # First dim of Hn is num_layers, which is set to 1 above.

        return out

"""## Train model for Some Epoches"""

# init. epochs
epoches = 11

house_lstm = LSTM(num_sensors = 6, hidden_units = 32).to(device)


optimizer = torch.optim.Adam(params = house_lstm.parameters(), lr = 1e-3)
house_lstm_train_loss, house_lstm_test_loss = [], []


torch.manual_seed(64)
torch.cuda.manual_seed(64)
for epoch in tqdm(range(epoches)):

  train_loss = train_loop(model = house_lstm, dataloader = train_dataloader,
                                    loss_fn = loss_fn, optimizer = optimizer, device = device)
  
  test_loss = test_loop(model = house_lstm, dataloader = test_dataloader,
                                  loss_fn = loss_fn, device = device)
  
  house_lstm_train_loss.append(train_loss.item())
  house_lstm_test_loss.append(test_loss.item())



  print(f"Epoch: {epoch+1}  Train Loss: {train_loss:.4f} | Test Loss: {test_loss:.4f}")

plot_graph(house_lstm_train_loss, house_lstm_test_loss)

gc.collect()
torch.cuda.empty_cache()

"""## real global active power and predicted global active power"""

model_val = eval_loop(model = house_lstm, dataloader = test_dataloader,
                                  loss_fn = loss_fn, device = device)

model_val = np.c_[model_val]
model_val.shape

plt.figure(figsize = (20, 12))
# plt.subplot(1, 2, 1)
plt.plot(range(len(y_test)), y_test, label = "Actual")
plt.plot(range(len(model_val)), model_val, label = "Predict")
plt.legend()
plt.title("Real global active power VS Predicted global active power")
plt.xlabel("number of values")
plt.ylabel("values")

gc.collect()
torch.cuda.empty_cache()

"""# Splite dataset into 70:30"""

info = info.astype(float)

info_numpy = info.to_numpy()
x = info_numpy[:,1:]
y = info_numpy[:,0:1]

print(x)
print()
print(y)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30, random_state=64)

"""### Now Apply standard scaller on train and test data"""

data_scaler = StandardScaler()
x_train = data_scaler.fit_transform(x_train)
x_test = data_scaler.transform(x_test)


# info_scaler = pd.DataFrame(info_scaler, columns=['Global_active_power','Global_reactive_power','Voltage','Global_intensity', 'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3'])
print(x_train)
print()
print(x_test)

pred_scaler = StandardScaler()
pred_scaler.fit(y_train)
y_train = pred_scaler.fit_transform(y_train)
y_test = pred_scaler.transform(y_test)


print(y_train)
print()
print(y_test)

"""## Make DataLoader"""

BATCH_SIZE = 64

train_dataloader = numpy_to_dataLoader(x_train, y_train, batch_size = BATCH_SIZE, shuffle = True)
test_dataloader = numpy_to_dataLoader(x_test, y_test, batch_size = BATCH_SIZE, shuffle = False)

"""## Train Model"""

# init. epochs
epoches = 11

house_lstm = LSTM(num_sensors = 6, hidden_units = 32).to(device)


optimizer = torch.optim.Adam(params = house_lstm.parameters(), lr = 1e-3)
house_lstm_train_loss, house_lstm_test_loss = [], []
house_lstm_train_accs, house_lstm_test_accs = [], []


torch.manual_seed(64)
torch.cuda.manual_seed(64)
for epoch in tqdm(range(epoches)):

  train_loss = train_loop(model = house_lstm, dataloader = train_dataloader,
                                    loss_fn = loss_fn, optimizer = optimizer, device = device)
  
  test_loss = test_loop(model = house_lstm, dataloader = test_dataloader,
                                  loss_fn = loss_fn, device = device)
  
  house_lstm_train_loss.append(train_loss.item())
  house_lstm_test_loss.append(test_loss.item())


  print(f"Epoch: {epoch+1}  Train Loss: {train_loss:.4f} | Test Loss: {test_loss:.4f}")

plot_graph(house_lstm_train_loss, house_lstm_test_loss)

gc.collect()
torch.cuda.empty_cache()

"""## real global active power and predicted global active power"""

model_val = eval_loop(model = house_lstm, dataloader = test_dataloader,
                                  loss_fn = loss_fn, device = device)

model_val = np.c_[model_val]
model_val.shape

plt.figure(figsize = (20, 12))
# plt.subplot(1, 2, 1)
plt.plot(range(len(y_test)), y_test, label = "Actual")
plt.plot(range(len(model_val)), model_val, label = "Predict")
plt.legend()
plt.title("Real global active power VS Predicted global active power")
plt.xlabel("number of values")
plt.ylabel("values")

gc.collect()
torch.cuda.empty_cache()