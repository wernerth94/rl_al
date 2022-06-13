import torch, sys

sys.path.append("../core")
sys.path.append("../rl_core")
print(F"updated path is {sys.path}")

from tqdm import tqdm
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, TensorDataset
from torch.nn import CrossEntropyLoss
from torch.optim import SGD
from torchvision import transforms
from torchvision.models.densenet import DenseNet
from core.Data import load_cifar10_pytorch

device = "cuda" if torch.cuda.is_available() else "cpu"

k = 12
d = 40
model = DenseNet(growth_rate=k, drop_rate=0.2, num_classes=10)
model = model.to(device)
optimizer = SGD(model.parameters(), lr=0.1)
loss = CrossEntropyLoss()

x_train, y_train, x_test, y_test = load_cifar10_pytorch(prefix="..")
x_train = x_train.to(device)
y_train = y_train.to(device)
x_test = x_test.to(device)
y_test = y_test.to(device)
preprocess = transforms.Compose([
    transforms.Resize(224),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
train_dataloader = DataLoader(TensorDataset(x_train, y_train), batch_size=64)
test_dataloader = DataLoader(TensorDataset(x_test, y_test), batch_size=64)

train_losses = list()
test_losses = list()
test_f1 = list()

MAX_EPOCHS = 40
for e in range(MAX_EPOCHS):
    if e == 20:
        for param_group in optimizer.param_groups:
            param_group['lr'] = 0.01
    if e == 30:
        for param_group in optimizer.param_groups:
            param_group['lr'] = 0.001

    epoch_loss = 0.0
    print(f"{e}/{MAX_EPOCHS} - train")
    iterator = tqdm(train_dataloader)
    i = 0
    for batch_x, batch_y in iterator:
        i += 1
        iterator.set_postfix(loss=epoch_loss/i, refresh=True)
        batch_x = preprocess(batch_x)
        yHat = model(batch_x)
        loss_value = loss(yHat, batch_y)
        optimizer.zero_grad()
        loss_value.backward()
        optimizer.step()
        epoch_loss += loss_value.detach().item()

    train_losses.append(epoch_loss)
    # early stopping on test
    print(f"{e}/{MAX_EPOCHS} - test")
    with torch.no_grad():
        epoch_loss = 0.0
        with tqdm(test_dataloader) as iterator:
            i = 0
            for batch_x, batch_y in iterator:
                i += 1
                iterator.set_postfix(loss=epoch_loss/i, refresh=True)
                batch_x = preprocess(batch_x)
                yHat_test = model(batch_x)
                test_loss = loss(yHat_test, batch_y)
                epoch_loss += test_loss.detach().item()
        if test_loss >= epoch_loss:
            # print(f"labeled {len(self.xLabeled)}: stopped after {e} epochs")
            break
        lastLoss = epoch_loss
        one_hot_y_hat = torch.eye(10)[torch.argmax(yHat_test, dim=1).cpu()]
        newTestF1 = f1_score(y_test.cpu(), one_hot_y_hat.cpu(), average="samples")

        test_losses.append(epoch_loss)
        test_f1.append(newTestF1)
        print(f"train_loss: {epoch_loss}, test_loss: {test_loss.item()}, f1: {newTestF1}")
