import torch 
import torch.nn as nn
import torch.nn.functional as F
import math
from torchvision import datasets, transforms, models
import numpy as np
import random
import matplotlib.pyplot as plt
from resnet import resnet32
import wandb
wandb.login(key= "3ab0ccc1e73901e7d78c5af7de65194191805602")
from collections import Counter

# from sklearn.metrics import classification_report

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
cifar10_train = datasets.CIFAR10('Datasets/CIFAR10/train', download=True, train=True, transform=transform_train)
cifar10_test = datasets.CIFAR10('Datasets/CIFAR10/test',download=True, train=False, transform=transform_test)

num_classes = float(len(cifar10_train.classes))
imbalance_ratio = 100.0
imbalance_factor = math.exp(-math.log(imbalance_ratio)/(num_classes-1))



num_samples_0 = cifar10_train.targets.count(0)
all_indexes_seperate = []
pi_list = []
for i in range(10): # 10 classes
  num_samples_i = int(num_samples_0 * (imbalance_factor)**(i))
  i_indexes = [k for k, j in enumerate(cifar10_train.targets) if j == i]
  sample_indexes = random.sample(i_indexes, num_samples_i)
  all_indexes_seperate.append(sample_indexes)
  pi_list.append(len(sample_indexes))

all_indexes = sum(all_indexes_seperate, []) # collapse list
sampled_images = cifar10_train.data[all_indexes] 
sampled_targets = np.array(cifar10_train.targets)[all_indexes]
pi_list_tensor = torch.tensor(pi_list) / len(all_indexes)
log_prob_tensor = torch.log(pi_list_tensor)
cifar10_train.data = sampled_images
cifar10_train.targets = list(sampled_targets)

sampler = torch.utils.data.WeightedRandomSampler(weights= (1/pi_list_tensor)**0.5, num_samples = 100, replacement = True)

trainloader = torch.utils.data.DataLoader(cifar10_train, batch_size=128,  num_workers=8, sampler=sampler)
testloader =  torch.utils.data.DataLoader(cifar10_test, batch_size=512, shuffle=True, num_workers=8)
model = resnet32()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum = 0.9, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones = [800, 950, 1100], gamma = 0.1)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

num_epochs = 1200
temp = 1.5


def ERM(trainloader, criterion, model, optimizer, scheduler, testloader):
  print('Vanilla training')
  for epoch in range(num_epochs):
      for i, (images, labels) in enumerate(trainloader):  
          # Move tensors to the configured device
          images = images.to(device)
          labels = labels.to(device)
          model.to(device)
          # Forward pass
          outputs = model(images)
          loss = criterion(outputs, labels)
          loss_r = 0
          for parameter in model.parameters():
            loss_r += torch.sum(parameter ** 2)
          loss = loss +  1e-4 * loss_r
          # Backward and optimize
          optimizer.zero_grad()
          loss.backward()
          optimizer.step()
          
      
      scheduler.step()
      wandb.log({"loss": loss})
      print ('ERM: Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, loss.item()))
      model.eval()
      test(testloader, model)
      test_posthoc(testloader, model, temp, log_prob_tensor)    
      model.train()

      
  # torch.save(model.state_dict(), 'vanilla') 

def test(testloader, model):
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

  wandb.log({'Vanilla Test Accuracy': 100 * correct / total})

def test_posthoc(testloader, model, temp, log_prob):
  correct = 0
  total = 0
  for images, labels in testloader:
      images = images.to(device)
      labels = labels.to(device)
      # model.to(device)
      with torch.no_grad():
        outputs = model(images)
      logits = outputs.data - (temp * log_prob).to(device) 
      _, predicted = torch.max(logits, 1)
      total += labels.size(0)
      correct += (predicted == labels).sum().item()
      torch.cuda.empty_cache()

  wandb.log({'Post-hoc adjusted Test Accuracy': 100 * correct / total})

class LA_Loss(nn.Module):
  def __init__(self, temp, log_prob):
    super().__init__()
    self.temp = temp
    self.log_prob = log_prob
  
  def forward(self, input, target):
    out = F.cross_entropy(input + self.temp * self.log_prob, target)
    return out

# adjusted CE loss
def adjusted_loss(trainloader, criterion, model, optimizer, scheduler, temp, log_prob_tensor):
  for epoch in range(num_epochs):
      for i, (images, labels) in enumerate(trainloader):  
          # Move tensors to the configured device
          images = images.to(device)
          labels = labels.to(device)
          model.to(device)
          # Forward pass
          outputs = model(images)
          criterion = LA_Loss(temp, log_prob_tensor.to(device))
          loss = criterion(outputs, labels)
          # Backward and optimize
          optimizer.zero_grad()
          loss.backward()
          optimizer.step()

      scheduler.step()    
      print ('Adjusted Loss Training: Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, loss.item()))
      test(testloader, model)

if(__name__ == '__main__'):
    # wandb.init(project="logit_adjustment_final", name="ERM with Sampler")
    print((1/pi_list_tensor)**0.25)
    # ERM(trainloader, criterion, model, optimizer, scheduler, testloader)
    # adjusted_loss_(trainloader, criterion, model, optimizer, scheduler, temp, log_prob_tensor)