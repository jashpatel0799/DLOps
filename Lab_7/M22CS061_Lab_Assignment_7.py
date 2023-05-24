import os
import torch
from torch import nn
from torchvision import datasets, transforms, utils
from torch.utils.data.dataloader import default_collate
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
import torch.utils.tensorboard as tb
from timeit import default_timer as timer
from itertools import product
from torchmetrics.classification import MulticlassAccuracy
from tqdm.auto import tqdm
import matplotlib.pyplot as plt


device = "cuda" if torch.cuda.is_available() else "cpu"
print(f'device: {device}')

train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    #transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    #transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

## Get Dataset
train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)
test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transform)

## Make DataLoader
BATCH_SIZE = 32
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=lambda x: tuple(x_.to(device) for x_ in default_collate(x)))
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=lambda x: tuple(x_.to(device) for x_ in default_collate(x)))


# train
def train_loop(model: torch.nn.Module, dataloader: torch.utils.data.DataLoader, 
               loss_fn: torch.nn.Module, optimizer: torch.optim.Optimizer, 
               accuracy_fn, device: torch.device = device):
  
  model.train()

  train_loss, train_acc = 0, 0

  for batch, (x_train, y_train) in enumerate(dataloader):

    if device == 'cuda':
      x_train, y_train = x_train.to(device), y_train.to(device)

    
    # 1. Forward
    y_pred = model(x_train)

    # 2. Loss
    loss = loss_fn(y_pred, y_train)
    acc = accuracy_fn(y_train, torch.argmax(y_pred, dim = 1))

    train_loss += loss
    train_acc += acc

    # 3. optimizer zero grad
    optimizer.zero_grad()

    # 4. Backward
    loss.backward()

    # 5. optimizer step
    optimizer.step()

  train_loss /= len(dataloader)
  train_acc /= len(dataloader)

  return train_loss, train_acc


def test_loop(model: torch.nn.Module, dataloader: torch.utils.data.DataLoader,
              loss_fn: torch.nn.Module, accuracy_fn, device: torch.device = device):
  test_loss, test_acc = 0, 0

  model.eval()
  with torch.inference_mode():
    for x_test, y_test in dataloader:
      if device == 'cuda':
        x_test, y_test = x_test.to(device), y_test.to(device)

      # 1. Forward
      test_pred = model(x_test)

      # 2. Loss
      test_loss += loss_fn(test_pred, y_test)
      test_acc += accuracy_fn(y_test, torch.argmax(test_pred, dim = 1))


    test_loss /= len(dataloader)
    test_acc /= len(dataloader)

  return test_loss, test_acc


# Loss
loss_fn = nn.CrossEntropyLoss()

# Accuracy
accuracy_fn = MulticlassAccuracy(num_classes = 10).to(device)


def plotplot(train_losses, test_losses, train_acces, test_acces, file_name):
  plt.figure(figsize = (25,8))
  plt.subplot(1,2,1)
  plt.plot(range(len(train_losses)),train_losses, label = "Train Loss")
  plt.plot(range(len(test_losses)),test_losses, label = "Test Loss")
  plt.xlabel("Epoches")
  plt.ylabel("Loss")
  plt.title("Loss vs Epoches")
  plt.legend()

  plt.subplot(1,2,2)
  plt.plot(range(len(train_acces)),train_acces, label = "Train Accuracy")
  plt.plot(range(len(test_acces)),test_acces, label = "Test Accuracy")
  plt.xlabel("Epoches")
  plt.ylabel("Accuracy")
  plt.title("Accuracy vs Epoches")
  plt.legend()

  plt.savefig(file_name)




# For 10 Epoches
torch.manual_seed(64)
torch.cuda.manual_seed(64)
mv2 = mobilenet_v2(weights = MobileNet_V2_Weights.DEFAULT).to(device)
mv2.classifier[1] = nn.Linear(1280, 10).to(device)
optimizer = torch.optim.Adam(mv2.parameters(), lr=0.001)


train_losses, test_losses = [], []
train_acces, test_acces = [], []

epoches = 10

torch.manual_seed(64)
torch.cuda.manual_seed(64)
start = timer() 

for epoch in tqdm(range(epoches)):

    print(f"Epoch: {epoch + 1}")
     
    train_loss, train_acc = train_loop(model = mv2, dataloader = train_dataloader,
                                        loss_fn = loss_fn, optimizer = optimizer, 
                                        accuracy_fn = accuracy_fn, device = device)
    test_loss, test_acc = test_loop(model = mv2, dataloader = test_dataloader,
                                    loss_fn = loss_fn, accuracy_fn = accuracy_fn,
                                    device = device)

    
    print(f"Train Loss: {train_loss:.4f} | Test Loss: {test_loss:.4f} || Train Accuracy: {train_acc:.4f} | Test Accuracy: {test_acc:.4f}")

    train_losses.append(train_loss.item())
    test_losses.append(test_loss.item())
    train_acces.append(train_acc.item())
    test_acces.append(test_acc.item())

end = timer()

print(f"Time To Train Model: {(end - start):.2f} seconds")

plotplot(train_losses, test_losses, train_acces, test_acces, "mv2_10.jpg")



# For 15 Epoches
lr = 1e-3
torch.manual_seed(64)
torch.cuda.manual_seed(64)
mv2 = mobilenet_v2(weights = MobileNet_V2_Weights.DEFAULT).to(device)
mv2.classifier[1] = nn.Linear(1280, 10).to(device)
optimizer = torch.optim.Adam(mv2.parameters(), lr=lr)


train_losses, test_losses = [], []
train_acces, test_acces = [], []

epoches = 15
torch.manual_seed(64)
torch.cuda.manual_seed(64)
start = timer() 

for epoch in tqdm(range(epoches)):

    print(f"Epoch: {epoch + 1}")
     
    train_loss, train_acc = train_loop(model = mv2, dataloader = train_dataloader,
                                        loss_fn = loss_fn, optimizer = optimizer, 
                                        accuracy_fn = accuracy_fn, device = device)
    test_loss, test_acc = test_loop(model = mv2, dataloader = test_dataloader,
                                    loss_fn = loss_fn, accuracy_fn = accuracy_fn,
                                    device = device)

    
    print(f"Train Loss: {train_loss:.4f} | Test Loss: {test_loss:.4f} || Train Accuracy: {train_acc:.4f} | Test Accuracy: {test_acc:.4f}")

    train_losses.append(train_loss.item())
    test_losses.append(test_loss.item())
    train_acces.append(train_acc.item())
    test_acces.append(test_acc.item())

end = timer()

print(f"Time To Train Model: {(end - start):.2f} seconds")

plotplot(train_losses, test_losses, train_acces, test_acces, "mv2_15.jpg")



# Hyperparameter
lr = [0.01, 0.001]
betas = [(0.85, 0.8585), (0.95, 0.9595)]

hyper_params = [(l, b) for l in lr for b in betas]

epoches = 5

cur_iter,total_iter = 1, len(lr)*len(betas)
for h_p in hyper_params:
    print()
    print(f"iter: {cur_iter} / {total_iter}")
    print(f"Tuning with --- lr: {h_p[0]}, betas: {h_p[1]}")
    train_loss_list, test_loss_list = [], []

    torch.manual_seed(64)
    torch.cuda.manual_seed(64)
    mv2 = mobilenet_v2(weights = MobileNet_V2_Weights.DEFAULT).to(device)
    mv2.classifier[1] = nn.Linear(1280, 10).to(device)
    optimizer = torch.optim.Adam(mv2.parameters(), lr=h_p[0], betas = h_p[1])


    train_losses, test_losses = [], []
    train_acces, test_acces = [], []

    torch.manual_seed(64)
    torch.cuda.manual_seed(64)
    start = timer() 
    for epoch in tqdm(range(epoches)):

        print(f"Epoch: {epoch + 1}")
        
        train_loss, train_acc = train_loop(model = mv2, dataloader = train_dataloader,
                                            loss_fn = loss_fn, optimizer = optimizer, 
                                            accuracy_fn = accuracy_fn, device = device)

        test_loss, test_acc = test_loop(model = mv2, dataloader = test_dataloader,
                                        loss_fn = loss_fn, accuracy_fn = accuracy_fn,
                                        device = device)

        
        print(f"Train Loss: {train_loss:.4f} | Test Loss: {test_loss:.4f} || Train Accuracy: {train_acc:.4f} | Test Accuracy: {test_acc:.4f}")

        train_losses.append(train_loss.item())
        test_losses.append(test_loss.item())
        train_acces.append(train_acc.item())
        test_acces.append(test_acc.item())

    end = timer()

    print(f"Time To Train Model: {(end - start):.2f} seconds")

    plotplot(train_losses, test_losses, train_acces, test_acces, f"mv2_hp{cur_iter}.jpg")
    cur_iter += 1