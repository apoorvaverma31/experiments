import torch 
import torch.nn as nn
import torch.nn.functional as F
import math
from torchvision import datasets, transforms, models
import numpy as np
import random
import matplotlib.pyplot as plt
from resnet32 import resnet32
from os import path
import wandb
wandb.login(key= "3ab0ccc1e73901e7d78c5af7de65194191805602")
from sklearn.metrics import f1_score, confusion_matrix
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
cifar10_train = datasets.CIFAR10('/workspace/Datasets/CIFAR10/train', download=False, train=True, transform=transform_train)

num_classes = float(len(cifar10_train.classes))
imbalance_ratio = 100.0
imbalance_factor = math.exp(-math.log(imbalance_ratio)/(num_classes-1))



num_samples_0 = cifar10_train.targets.count(0)
all_indexes_seperate = []
pi_list = []
num_samples_class = []

for i in range(10): # 10 classes
  num_samples_i = int(num_samples_0 * (imbalance_factor)**(i))
  num_samples_class.append(num_samples_i)
  i_indexes = [k for k, j in enumerate(cifar10_train.targets) if j == i]
  sample_indexes = random.sample(i_indexes, num_samples_i)
  all_indexes_seperate.append(sample_indexes)
  pi_list.append(len(sample_indexes))

all_indexes = sum(all_indexes_seperate, []) # collapse list
sampled_images = cifar10_train.data[all_indexes] 
sampled_targets = np.array(cifar10_train.targets)[all_indexes]
pi_list_tensor = torch.tensor(pi_list) / len(all_indexes)
flipped=pi_list_tensor.flip((0,))
a = np.divide(pi_list_tensor, np.array(num_samples_class))
log_prob_tensor = torch.log(pi_list_tensor)
cifar10_train.data = sampled_images
cifar10_train.targets = list(sampled_targets)

y_train = [cifar10_train.targets[i] for i in np.arange(len(cifar10_train.targets))]
b = np.array([a[t] for t in y_train])
sampler = torch.utils.data.WeightedRandomSampler(weights= b, num_samples = len(b), replacement = True)

trainloader = torch.utils.data.DataLoader(cifar10_train, batch_size = 128,  num_workers=8, shuffle=True)
model = resnet32()
# model = nn.DataParallel(model)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum = 0.9, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones = [120, 135], gamma = 0.1)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

num_epochs = 200
temp = 1.5
# def make_lt_dataset(train_data, imbalance_ratio, inv=False):
#     pi_list = []
#     num_samples_0 = train_data.targets.count(0)
#     num_classes = float(len(train_data.classes))
#     imbalance_factor = math.exp(-math.log(imbalance_ratio)/(num_classes-1))
#     all_indexes_seperate = []
#     for i in range(10): # 10 
#         i_indexes = [k for k, j in enumerate(train_data.targets) if j == i]
#         if(inv):
#           num_samples_i = int(num_samples_0 * (imbalance_factor)**(9-i))
#         else:
#           num_samples_i = int(num_samples_0 * (imbalance_factor)**(i))
#         sample_indexes = random.sample(i_indexes, num_samples_i)
#         pi_list.append(len(sample_indexes))
#         all_indexes_seperate.append(sample_indexes)
        
#     all_indexes = sum(all_indexes_seperate, []) # collapse list
#     pi_list_tensor = torch.tensor(pi_list) / len(all_indexes)
#     sampled_images = train_data.data[all_indexes] 
#     sampled_targets = np.array(train_data.targets)[all_indexes]
#     train_data.data = sampled_images
#     train_data.targets = list(sampled_targets)
#     return train_data, pi_list_tensor
cifar10_test = datasets.CIFAR10('/workspace/Datasets/CIFAR10/train',download=False, train=False, transform=transform_test)
# cifar10_test = make_lt_dataset(cifar10_test, 100, inv=True)
testloader =  torch.utils.data.DataLoader(cifar10_test, batch_size=512, shuffle=True, num_workers=8, drop_last=True)


def ERM(trainloader, criterion, model, optimizer, scheduler, testloader):
  print('Vanilla training')
  for epoch in range(num_epochs):
      for i, (images, labels) in enumerate(trainloader):  
          # Move tensors to the configured device
          images = images.to(device)
          labels = labels.to(device)
          # print(labels)
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
          # print('loss:', loss)
          
      
      scheduler.step()
      wandb.log({"loss": loss, 'epoch':epoch})
      print ('ERM: Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, loss.item()))
      model.eval()
      test(testloader, model, epoch)
      # test_posthoc(testloader, model, temp, log_prob_tensor, epoch)    
      model.train()

      
  # torch.save(model.state_dict(), 'vanilla') 

def test(testloader, model, epoch):
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

      _, predicted = torch.max(outputs.data, 1)
      total += labels.size(0)
      correct += (predicted == labels).sum().item()
      preds.append(predicted.detach().cpu().numpy())
      truths.append(labels.detach().cpu().numpy())
      torch.cuda.empty_cache()

  truths = np.array(truths).reshape(-1)
  preds = np.array(preds).reshape(-1)
  cm = confusion_matrix(list(truths), list(preds))

  wandb.log({'Vanilla Test Accuracy': 100 * correct / total, 'epoch':epoch})
  print('f1_score: ',f1_score(truths, preds, average=None))
  print('Class-wise accuracy',cm.diagonal()/cm.sum(axis=1))
  print('Accuracy:', 100*correct/total)

def test_posthoc(testloader, model, temp, log_prob, epoch):
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

  wandb.log({'Post-hoc adjusted Test Accuracy': 100 * correct / total, 'epoch':epoch})

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

def plot(X, y, title, x_label, y_label):
  plt.title(title)
  plt.xlabel(x_label)
  plt.ylabel(y_label)
  plt.plot(X, y, color ="red")
  plt.show(block=True)


if(__name__ == '__main__'):
    wandb.init(project="differential_tail", name="PostHoc_Balanced_Test", mode="disabled")
    wandb.watch(model, log='all')
    # ERM(trainloader, criterion, model, optimizer, scheduler, testloader)
    # torch.save(model.state_dict(), 'checkpoint')
    model.load_state_dict(torch.load('checkpoint'))
    print('model loaded')
    model.eval()
    test(testloader, model, 1)
    print('classifer normalized')
    model.classifier_weight_norm(2)
    test(testloader, model, 1)