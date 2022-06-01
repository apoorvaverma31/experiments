from pyexpat import model
import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import datasets, transforms, models
from resnet32 import resnet32
import random
from sklearn.metrics import f1_score, confusion_matrix
import numpy as np
import wandb
import math
from collections import Counter
wandb.login(key= "3ab0ccc1e73901e7d78c5af7de65194191805602")
wandb.init(project="differential_tail", name="BalancedSampling_Inv_LT_Test", config={"num_epochs":200, "batch_size":128,
         "lr":0.1, "momentum":0.9, "weight_decay":0.0001}) # set mode="disabled" to not track logs
config = wandb.config



def make_lt_dataset(train_data, imbalance_ratio, inv=False):
    pi_list = []
    num_samples_0 = train_data.targets.count(0)
    num_classes = float(len(train_data.classes))
    imbalance_factor = math.exp(-math.log(imbalance_ratio)/(num_classes-1))
    all_indexes_seperate = []
    for i in range(10): # 10 
        i_indexes = [k for k, j in enumerate(train_data.targets) if j == i]
        if(inv):
          num_samples_i = int(num_samples_0 * (imbalance_factor)**(9-i))
        else:
          num_samples_i = int(num_samples_0 * (imbalance_factor)**(i))
        sample_indexes = random.sample(i_indexes, num_samples_i)
        pi_list.append(len(sample_indexes))
        all_indexes_seperate.append(sample_indexes)
        
    all_indexes = sum(all_indexes_seperate, []) # collapse list
    pi_list_tensor = torch.tensor(pi_list) / len(all_indexes)
    sampled_images = train_data.data[all_indexes] 
    sampled_targets = np.array(train_data.targets)[all_indexes]
    train_data.data = sampled_images
    train_data.targets = list(sampled_targets)
    return train_data, pi_list_tensor


def get_loaders(train_data, test_data, sampler=None):
  if(sampler is not None):
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=config.batch_size, sampler=sampler,  num_workers=8)
  
  else:
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=config.batch_size, shuffle=True,  num_workers=8)
  testloader =  torch.utils.data.DataLoader(test_data, batch_size=4 * config.batch_size, shuffle=True, num_workers=8, drop_last=True)
  return trainloader, testloader

def get_sampler(train_data, pi_tensor):
  y_train = [train_data.targets[i] for i in np.arange(len(train_data.targets))]
  b = []
  for t in y_train:
    b.append(pi_tensor[t])
  sample_wts = np.array(b)

  sampler = torch.utils.data.WeightedRandomSampler(weights= 1/sample_wts, num_samples = len(b), replacement = True)
  return sampler

def train(model, trainloader, criterion, optimizer, scheduler, device, testloader, num_epochs, pi_list=None, temp=None):
  for epoch in range(num_epochs):
    for images, labels in trainloader:
      images, labels = images.to(device), labels.to(device)
      model.to(device)

      outputs = model(images)
      loss = criterion(outputs, labels)

      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
    
    scheduler.step()
    wandb.log({"Training Loss":loss, 'Epoch':epoch})
    print ('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, loss.item()))
    model.eval()
    test(model, device, testloader, epoch, pi_list, temp)
    model.train()

def test(model, device, testloader, epoch, pi_list=None, temp=None):
  correct = 0
  total = 0
  preds = []
  truths = []
  for images, labels in testloader:
      images = images.to(device)
      labels = labels.to(device)
      model.to(device)
      with torch.no_grad():
        outputs = model(images)
      
      if(pi_list is not None and temp is not None):
        logits = outputs.data - (temp * torch.log(pi_list)).to(device) 


      _, predicted = torch.max(logits, 1)
      total += labels.size(0)
      correct += (predicted == labels).sum().item()
      preds.append(predicted.detach().cpu().numpy())
      truths.append(labels.detach().cpu().numpy())
      torch.cuda.empty_cache()

  truths = np.array(truths).reshape(-1)
  preds = np.array(preds).reshape(-1)
  cm = confusion_matrix(list(truths), list(preds))
  classwise_accuracy = cm.diagonal()/cm.sum(axis=1)

  wandb.log({'Test Accuracy': 100 * correct / total, 'Epoch': epoch})
  wandb.log({'Head Classes Accuracy': 100 * np.average(classwise_accuracy[0:4]), 'Epoch': epoch})
  wandb.log({'Mid Classes Accuracy': 100 * np.average(classwise_accuracy[4:7]), 'Epoch': epoch})
  wandb.log({'Tail Classes Accuracy': 100 * np.average(classwise_accuracy[7:10]), 'Epoch': epoch})
  
  print('Test Accuracy: ', 100 * correct / total)
  print('Head Classes Accuracy:', 100 * np.average(classwise_accuracy[0:4]))
  print('Mid Classes Accuracy:', 100 * np.average(classwise_accuracy[4:7]))
  print('Tail Classes Accuracy:', 100 * np.average(classwise_accuracy[7:10]))
  print('Classwise Accuracy: ', classwise_accuracy)
  # print('f1 score: ',f1_score(truths, preds, average=None))


def main():

    transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    cifar10_train = datasets.CIFAR10('/workspace/Datasets/CIFAR10/train', download=False, train=True, transform=transform_train)
    cifar10_test = datasets.CIFAR10('/workspace/Datasets/CIFAR10/test',download=False, train=False, transform=transform_test)
    cifar10_train, pi_tensor = make_lt_dataset(cifar10_train, 100)
    cifar10_test, _ = make_lt_dataset(cifar10_test, 100, inv=True)
    sampler = get_sampler(cifar10_train, pi_tensor)
    y_sampled = [cifar10_train.targets[i] for i in list(sampler)]
    trainloader, testloader = get_loaders(cifar10_train, cifar10_test, sampler)
    model = resnet32()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=config.lr, momentum = config.momentum, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones = [150, 180], gamma = 0.1)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    wandb.watch(model, log='all')
    train(model, trainloader, criterion, optimizer, scheduler, device, testloader, config.num_epochs, pi_tensor, 1.5)
   
if (__name__=="__main__"):
  main()
  


