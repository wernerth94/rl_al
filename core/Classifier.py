import torch.nn as nn
import torch
import torchvision
import torchvision.models as models
from torchvision import transforms

class Cifar10ClassifierFactory:
    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, numClasses=10, *args, **kwargs):
        # model = models.resnet18()
        # num_features = model.fc.in_features
        # model.fc = nn.Linear(num_features, 10)
        # model.train()
        model = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2),
            nn.Dropout(0.2),
            nn.Conv2d(16, 32, kernel_size=3, stride=2),
            nn.Dropout(0.2),
            nn.Conv2d(32, 64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 128, kernel_size=3),
            nn.BatchNorm2d(128),
            nn.Flatten(),
            nn.Linear(1152, 256),
            nn.ReLU(),
            nn.Linear(256, 10)
        )
        return model


class EmbeddingClassifierFactory:
    def __init__(self, embeddingSize):
        self.embeddingSize = embeddingSize



    def __call__(self, numClasses=10, *args, **kwargs):
        model = nn.Sequential(nn.Linear(self.embeddingSize, 50),
                              nn.LeakyReLU(),
                              nn.Linear(50, numClasses),
                              nn.Softmax(dim=1))
        example_forward_input = torch.rand(1, self.embeddingSize)
        model = torch.jit.trace_module(model, { 'forward' : example_forward_input} )
        return model


if __name__ == '__main__':
    import numpy as np
    from sklearn.metrics import f1_score
    import Data
    from torch.utils.data import TensorDataset, DataLoader
    import torch.optim as optim
    # img_size = 224
    img_size = 32
    trans = transforms.Compose([transforms.Resize([img_size, img_size]),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                     std=[0.229, 0.224, 0.225])])
    trainset = torchvision.datasets.CIFAR10(root='../../datasets', train=True, download=True, transform=trans)
    testset = torchvision.datasets.CIFAR10(root='../../datasets', train=False, download=True, transform=trans)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)
    testloader = torch.utils.data.DataLoader(testset, batch_size=256, shuffle=False, num_workers=2)
    model = get_cifar10_classifier()
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())

    mov = 0.7
    loss = 0.0
    for i in range(100):
        b = 0
        for batch_x, batch_y in trainloader:
            b += 1
            yHat = model(batch_x)
            loss_value = loss_function(yHat, batch_y)
            optimizer.zero_grad()
            loss_value.backward()
            optimizer.step()
            # print(f"batch {b}/{int(len(trainset)/128)}\t loss: {loss_value.detach().numpy()}")
        with torch.no_grad():
            l = 0.0
            f1 = 0.0
            b = 0
            for batch_x, batch_y in testloader:
                b += 1.0
                # print(f"batch {b}/{int(len(testset)/128)}")
                yHat_test = model(batch_x)
                l += loss_function(yHat_test, batch_y).numpy()
                one_hot_y_hat = np.eye(10, dtype='uint8')[torch.argmax(yHat_test, dim=1)]
                one_hot_y_batch = np.eye(10, dtype='uint8')[batch_y]
                f1 += f1_score(one_hot_y_batch, one_hot_y_hat, average="samples")
        print(f'### test loss: {l/b} \t f1 score: {f1/b}')


# hidden: 10            f1: 0.941
# hidden: 0             f1: 0.942
# hidden: 100           f1: 0.943