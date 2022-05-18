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
wandb.login()
from tqdm import tqdm
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
cifar10_train = datasets.CIFAR10('~/Datasets/CIFAR10/train', download=True, train=True, transform=transform_train)
cifar10_test = datasets.CIFAR10('~/Datasets/CIFAR10/test',download=True, train=False, transform=transform_test)

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

config = dict(
        epochs = 200,
        classes = 10,
        batch_size = 128,
        lr = 0.1,
        imbalance_ratio = 100,
        training_method = "vanilla",
        testing_method = "vanilla")

model_res = resnet32()
loss = nn.CrossEntropyLoss()


def model_pipeline(hyperparameters):

    # tell wandb to get started
    with wandb.init(project="logit-adjustment", config=hyperparameters):
      # access all HPs through wandb.config, so logging matches execution!
      config = wandb.config

      # make the model, data, and optimization problem
      model, train_loader, test_loader, criterion, optimizer, scheduler = make(config)
      # print(model)

      # and use them to train the model
      train(model, train_loader, criterion, optimizer, scheduler, config)

      # and test its final performance
      test(model, test_loader)

    return model

def make(config):
    # Make the data
    # train, test = get_data(train=True), get_data(train=False)
    train, test = cifar10_train, cifar10_test
    train_loader = torch.utils.data.DataLoader(train, batch_size=config.batch_size, shuffle = True)
    test_loader = torch.utils.data.DataLoader(test, batch_size=config.batch_size, shuffle = True)

    # Make the model
    model = model_res

    # Make the loss and optimizer
    criterion = loss
    optimizer = torch.optim.SGD(model.parameters(), lr=config.lr, momentum = 0.9, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones = [160, 180], gamma = 0.1)
    
    return model, train_loader, test_loader, criterion, optimizer, scheduler

def train(model, loader, criterion, optimizer, scheduler, config):
    # Tell wandb to watch what the model gets up to: gradients, weights, and more!
    wandb.watch(model, criterion, log="all", log_freq=10)

    # Run training and track with wandb
    total_batches = len(loader) * config.epochs
    example_ct = 0  # number of examples seen
    batch_ct = 0
    for epoch in tqdm(range(config.epochs)):
        for _, (images, labels) in enumerate(loader):

            loss = train_batch(images, labels, model, optimizer, criterion)
            example_ct +=  len(images)
            batch_ct += 1

            # Report metrics every 25th batch
            if ((batch_ct + 1) % 25) == 0:
                train_log(loss, example_ct, epoch)
        scheduler.step()
        test(model, testloader)



def train_batch(images, labels, model, optimizer, criterion):
    images, labels = images.to(device), labels.to(device)
    model.to(device)
    
    # Forward pass ➡
    outputs = model(images)
    loss = criterion(outputs, labels)
    
    # Backward pass ⬅
    optimizer.zero_grad()
    loss.backward()

    # Step with optimizer
    optimizer.step()

    return loss

def train_log(loss, example_ct, epoch):
    # Where the magic happens
    wandb.log({"epoch": epoch, "loss": loss}, step=example_ct)
    print(f"Loss after " + str(example_ct).zfill(5) + f" examples: {loss:.3f}")

def test(model, test_loader):
    model.eval()

    # Run the model on some test examples
    with torch.no_grad():
        correct, total = 0, 0
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print(f"Accuracy of the model on the {total} " +
              f"test images: {100 * correct / total}%")
        
        wandb.log({"test_accuracy": correct / total})

    # Save the model in the exchangeable ONNX format
    # torch.onnx.export(model, images, "model.onnx")
    # wandb.save("model.onnx")

trainloader = torch.utils.data.DataLoader(cifar10_train, batch_size=16, shuffle=True)
testloader =  torch.utils.data.DataLoader(cifar10_test, batch_size=64, shuffle=True)
model = resnet32()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum = 0.9, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones = [160, 180], gamma = 0.1)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

num_epochs = 200
model = model_pipeline(config)

def Vanilla_training(trainloader, criterion, model, optimizer, scheduler):
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
          # Backward and optimize
          optimizer.zero_grad()
          loss.backward()
          optimizer.step()

      scheduler.step()    
      print ('Vanilla Training: Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, loss.item()))
  torch.save(model.state_dict(), 'vanilla')

temp = 1.5

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

  print('Vanilla Test Accuracy: {} %'.format(100 * correct / total))

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

  print('Post-hoc Modification Accuracy: {} %'.format(100 * correct / total))

class LA_Loss(nn.Module):
  def __init__(self, temp, log_prob):
    super().__init__()
    self.temp = temp
    self.log_prob = log_prob
  
  def forward(self, input, target):
    out = F.cross_entropy(input + self.temp * self.log_prob, target)
    return out

# adjusted CE loss
def adjusted_loss_training(trainloader, criterion, model, optimizer, scheduler, temp, log_prob_tensor):
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

      # scheduler.step()    
      print ('Adjusted Loss Training: Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, loss.item()))



# Vanilla_training(trainloader, criterion, model, optimizer, scheduler)
# test(testloader, model)
# test_posthoc(testloader, model, temp, log_prob_tensor)
# adjusted_loss_training(trainloader, criterion, model, optimizer, scheduler, temp, log_prob_tensor)
# test(testloader, model)