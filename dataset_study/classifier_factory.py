import numpy as np
import torch
import torch.nn as nn





class Net(nn.Module):
    def __init__(self, input_features, classes):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_features, 5*input_features)  # 5*5 from image dimension
        self.fc2 = nn.Linear(5*input_features, classes)
        self.relu = nn.ReLU(inplace=True)
        self.net = nn.Sequential(self.fc1,self.relu,self.fc2)
    def forward(self,x):
        return self.net(x)

def get_dataset_info(x,y):
    instances, features = x.shape
    classes = len(np.unique(y))
    print(f"Instances: {instances}, features: {features}, classes: {classes}")
    return instances, features, classes

def create_classifier(x_train, y_train)->nn.Module:
    # this should build and return a model based on the training data (# features, # classes, etc)
    # input in NUMPY ARRAY
    # return torch module
    instances, features, classes = get_dataset_info(x_train, y_train)

    model = Net(features,classes)
    return model