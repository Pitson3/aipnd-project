# Imports here
import torch
from torch import nn
from torch import optim
from torchvision import datasets, transforms, models
import time
from collections import OrderedDict
import matplotlib.pyplot as plt
import torch.nn.functional as F
import numpy as np


# Build the network
class Network(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size,
                 dropIn_p=0, dropHidden_p=0):
        super().__init__()
        # hidden layers tensors
        self.hidden_layers = nn.ModuleList(
            [nn.Linear(input_size, hidden_sizes[0])])
        pair_hidden_sizes = zip(hidden_sizes[:-1], hidden_sizes[1:])
        self.hidden_layers.extend([nn.Linear(h1, h2)
                                   for h1, h2 in pair_hidden_sizes])
        # outout layer tensor
        self.output = nn.Linear(hidden_sizes[-1], output_size)
        # dropout layer for input
        self.dropoutIn = nn.Dropout(p=dropIn_p)
        # dropout layer for each hidden layer
        self.dropoutHidden = nn.Dropout(p=dropHidden_p)

    def forward(self, x):
        # Apply the in-drop layer on input layer
        x = self.dropoutIn(x)
        # Apply linear compinations and activation functions and hidden-drop layer
        for hidden_layer in self.hidden_layers:
            x = hidden_layer(x)
            x = F.relu(x)
            x = self.dropoutHidden(x)
        # Apply the linear combination on the last hidden layer to get the logits (scores of output)
        x = self.output(x)
        # Apply good activation function to get high precision instead of softmax
        x = F.log_softmax(x, dim=1)
        return x


# Get classifier based on feature detectors (pre_trainned CNN model)
def get_classifier(pre_trained_model, hidden_sizes, output_size, dropIn_p=0, dropHidden_p=0):
    # freeze the pre_trained parameters
    for param in pre_trained_model.parameters():
        param.requires_grad = False
    input_size = pre_trained_model.classifier[0].state_dict()[
        'weight'].shape[1]
    print(f"input_size of features to the classifier: {input_size}")
    print(f'hidden_sizes in classifier: {hidden_sizes}')
    print(f"output_size of classes from the classifier: {output_size}")
    print()
    classifier = Network(input_size, hidden_sizes,
                         output_size, dropIn_p, dropHidden_p)
    pre_trained_model.classifier = classifier
    return pre_trained_model


# Build validation function
def validation(model, criterion, valid_loader, gpu):
    valid_loss = 0
    valid_accuracy = 0
    device = torch.device(
        'cuda:0' if torch.cuda.is_available() and gpu else 'cpu')
    print(f'validation using device:{device}')
    model.to(device)
    model.eval()
    # Iterate over batches
    for images, labels in valid_loader:
        images, labels = images.to(device), labels.to(device)
        # forward pass
        with torch.no_grad():
            outputs = model.forward(images)
        # calculate loss
        loss = criterion(outputs, labels)
        valid_loss += loss.item()
        ps = torch.exp(outputs)
        equality = (labels == ps.max(dim=1)[1])
        valid_accuracy += equality.type(torch.float64).mean().item()
    info = {'loss': valid_loss / len(valid_loader),
            'accuracy': valid_accuracy / len(valid_loader)}
    return info


# define Train the model function
def train(model, optimizer, criterion, trainloader, validloader, gpu, epochs=2, print_every=40):

    device = torch.device(
        'cuda:0' if torch.cuda.is_available() and gpu else 'cpu')
    print(f'train using device:{device}')
    model.to(device)
    model.train()
    running_loss = 0
    steps = 0
    start = time.time()
    print('Training started...')
    for e in range(epochs):
        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)
            # Clears the gradients of all optimized
            optimizer.zero_grad()
            # Forward pass
            outputs = model.forward(images)
            # Calculate loss error for training
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            steps += 1
            # Computes the gradient of current tensor
            loss.backward()
            # Performs a single optimization step.
            optimizer.step()
            if steps % print_every == 0:
                model.eval()
                valid = validation(model, criterion, validloader, gpu)
                train_loss = running_loss / print_every

                print(f'epoch {e+1}/{epochs}')
                print(f'trainning loss = {train_loss :0.4}')
                print(f'valid loss = {valid["loss"] :0.4} ...',
                      f'valid accuracy = {valid["accuracy"] :0.4}')
                print('.............................')
                running_loss = 0
                model.train()
    time_elapsed = time.time() - start

    print("\nTotal time: {:.0f}m {:.0f}s".format(
        time_elapsed//60, time_elapsed % 60))
    print('Training Finished...')

    measurements = {'train_loss': train_loss,
                    'valid_loss': valid["loss"], 'valid_accuracy': valid["accuracy"]}
    return measurements
