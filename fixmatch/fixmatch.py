from numpy import False_
import torch 
import torch.nn as nn
import torch.nn.functional as F
import math
from torchvision import datasets, transforms, models
from wideresnet import build_wideresnet

import wandb
wandb.login(key= "3ab0ccc1e73901e7d78c5af7de65194191805602")
wandb.init(project="fixmatch", name="ERM_supervised", config={"num_epochs":200, "batch_size":64, "lr":0.03, "momentum":0.9, "weight_decay":0.005, "pl_threshold":0.95,        "l_ul_ratio":7,
          "uloss_wt":1}) # set mode="disabled" to not track logs
config = wandb.config


transform_train = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])
cifar10_train = datasets.CIFAR10('/workspace/Datasets/CIFAR10/train', download=False, train=True, transform=transform_train)
cifar10_test = datasets.CIFAR10('/workspace/Datasets/CIFAR10/test',download=False, train=False, transform=transform_test)

def get_loaders(train_data, test_data):
  trainloader = torch.utils.data.DataLoader(train_data, batch_size=config.batch_size,  num_workers=8)
  testloader =  torch.utils.data.DataLoader(test_data, batch_size=512, shuffle=True, num_workers=8)
  return trainloader, testloader

def train(model, trainloader, criterion, optimizer, scheduler, device, testloader):
  for epoch in range(config.num_epochs):
    for images, labels in trainloader:
      images, labels = images.to(device), labels.to(device)
      model.to(device)

      outputs = model(images)
      loss = criterion(outputs, labels)

      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
    
    scheduler.step()
    wandb.log({"Loss":loss})
    print ('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, config.num_epochs, loss.item()))
    model.eval()
    test(model, device, testloader)
    model.train()

def test(model, device, testloader):
  correct = 0
  total = 0
  for images, labels in testloader:
      images = images.to(device)
      labels = labels.to(device)
      model.to(device)
      with torch.no_grad():
        outputs = model(images)

      _, predicted = torch.max(outputs.data, 1)
      total += labels.size(0)
      correct += (predicted == labels).sum().item()
      torch.cuda.empty_cache()

  wandb.log({'Test Accuracy': 100 * correct / total})

def main():
  trainloader, testloader = get_loaders(cifar10_train, cifar10_test)
  model = build_wideresnet(28, 2, 0, 10)
  criterion = nn.CrossEntropyLoss()
  optimizer = torch.optim.SGD(model.parameters(), lr=config.lr, momentum = config.momentum, weight_decay=config.weight_decay)
  scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, end_factor = 0.01)
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  train(model, trainloader, criterion, optimizer, scheduler, device, testloader)

if (__name__=="__main__"):
  main()