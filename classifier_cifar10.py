from torchsummary import summary
from cifar_net import TCNN, TCNN1, TCNN2, TCNN3, TCNN3_1, TCNN3_2, TCNN4, TCNN5, TCNN8, TCNN13, TCNN14, TCNN15, TCNN16
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime as dt

import torch
from torch import optim, nn
from torch.utils.data import DataLoader, TensorDataset, Dataset
from torchvision.utils import make_grid
from torchvision import transforms as T
from torchvision import models, datasets

from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import Accuracy, Loss, Precision, Recall
from ignite.handlers import LRScheduler, ModelCheckpoint, global_step_from_engine
from ignite.contrib.handlers import ProgressBar, TensorboardLogger
import ignite.contrib.engines.common as common

import os
from random import randint

class Net(nn.Module):
    def __init__(self, device):
        super(Net, self).__init__()     
        self.model = TCNN3_2(device) #M

        #self.model = models.mobilenet_v3_small() #2.5M
        #self.model.classifier[3] = nn.Linear(1024, 10)

    def forward(self, x):
        x=self.model(x)
        return x

def generate_dataloader(data, name, transform):
    if data is None: 
        return None
    
    # Read image files to pytorch dataset using ImageFolder, a generic data 
    # loader where images are in format root/label/filename
    # See https://pytorch.org/vision/stable/datasets.html
    if transform is None:
        dataset = datasets.ImageFolder(data, transform=T.ToTensor())
    else:
        dataset = datasets.ImageFolder(data, transform=transform)

    # Set options for device
    if use_cuda:
        kwargs = {"pin_memory": True, "num_workers": 8}
    else:
        kwargs = {}
    
    # Wrap image dataset (defined above) in dataloader 
    dataloader = DataLoader(dataset, batch_size=batch_size, 
                        shuffle=(name=="train"), 
                        **kwargs)
    
    return dataloader

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:1" if use_cuda else "cpu")

preprocess_transform_pretrain = T.Compose([
                T.ToTensor(),  # Converting cropped images to tensors
])

# Define batch size for DataLoaders
batch_size = 64

# Create DataLoaders for pre-trained models (normalized based on specific requirements)
#train_loader_pretrain = generate_dataloader(TRAIN_DIR, "train", transform=preprocess_transform_pretrain)
#val_loader_pretrain = generate_dataloader(val_img_dir, "val", transform=preprocess_transform_pretrain)

trainset = datasets.CIFAR10(root='./data', train=True, download=False, transform=preprocess_transform_pretrain)
train_loader_pretrain = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
testset = datasets.CIFAR10(root='./data', train=False, download=False, transform=preprocess_transform_pretrain)
val_loader_pretrain = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

model = Net(device)
model = model.to(device)
#summary(model, (3, 32, 32), batch_size=16)

# Define hyperparameters and settings
lr = 0.001  # Learning rate
num_epochs = 128  # Number of epochs
log_interval = 300  # Number of iterations before logging

# Set loss function (categorical Cross Entropy Loss)
loss_func = nn.CrossEntropyLoss()

# Set optimizer (using Adam as default)
optimizer = optim.Adam(model.parameters(), lr=lr)


# Setup pytorch-ignite trainer engine
trainer = create_supervised_trainer(model, optimizer, loss_func, device=device)

# Add progress bar to monitor model training
ProgressBar(persist=True).attach(trainer, output_transform=lambda x: {"Batch Loss": x})

# Define evaluation metrics
metrics = {
    "accuracy": Accuracy(), 
    "loss": Loss(loss_func),
}

# Evaluator for training data
train_evaluator = create_supervised_evaluator(model, metrics=metrics, device=device)

# Evaluator for validation data
evaluator = create_supervised_evaluator(model, metrics=metrics, device=device)

# Display message to indicate start of training
@trainer.on(Events.STARTED)
def start_message():
    print("Begin training")

# Log results from every batch
@trainer.on(Events.ITERATION_COMPLETED(every=log_interval))
def log_batch(trainer):
    batch = (trainer.state.iteration - 1) % trainer.state.epoch_length + 1
    print(f"Epoch {trainer.state.epoch} / {num_epochs}, "
          f"Batch {batch} / {trainer.state.epoch_length}: "
          f"Loss: {trainer.state.output:.3f}")

# Evaluate and print training set metrics
@trainer.on(Events.EPOCH_COMPLETED)
def log_training_loss(trainer):
    print(f"Epoch [{trainer.state.epoch}] - Loss: {trainer.state.output:.2f}")
    train_evaluator.run(train_loader_pretrain)
    epoch = trainer.state.epoch
    metrics = train_evaluator.state.metrics
    print(f"Train - Loss: {metrics['loss']:.3f}, "
          f"Accuracy: {metrics['accuracy']:.3f} ")

# Evaluate and print validation set metrics
@trainer.on(Events.EPOCH_COMPLETED)
def log_validation_loss(trainer):
    evaluator.run(val_loader_pretrain)
    epoch = trainer.state.epoch
    metrics = evaluator.state.metrics
    print(f"Validation - Loss: {metrics['loss']:.3f}, "
          f"Accuracy: {metrics['accuracy']:.3f}")

# Sets up checkpoint handler to save best n model(s) based on validation accuracy metric
common.save_best_model_by_val_score(
          output_path="best_models",
          evaluator=evaluator, model=model,
          metric_name="accuracy", n_saved=1,
          trainer=trainer, tag="val")

trainer.run(train_loader_pretrain, max_epochs=num_epochs)
print(evaluator.state.metrics)

