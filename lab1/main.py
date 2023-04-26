import os
import torch
import random
import numpy as np
from torch import nn
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from sklearn.metrics import confusion_matrix, classification_report

def fix_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 2048),
            nn.ReLU(),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 10)
        )
    def forward(self, x):
        x = self.model(x)
        return x


epoch = 5
lr = 1e-3
batch_size = 32
load = None # './model/model.pt' # 若不为None，表示使用之前训练过的模型

if __name__ == '__main__':
    fix_seed(1024)
    train_data = datasets.MNIST(root='./data/', transform=transforms.ToTensor(), train=True, download=True)
    test_data = datasets.MNIST(root='./data/', transform=transforms.ToTensor(), train=False, download=True)

    train_data, dev_data = random_split(train_data, [0.8, 0.2]) # 训练/验证集

    train_loader = DataLoader(train_data, batch_size=32)
    dev_loader = DataLoader(dev_data, batch_size=32)
    test_loader = DataLoader(test_data, batch_size=32)

    if load and os.path.exists(load):
        model = torch.load(load)
    else:
        model = MLP()
    model.to('cuda')
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    train_losses = []
    dev_losses = []
    for i in range(epoch):
        model.train()
        train_loss, dev_loss = 0, 0
        for X, y in tqdm(train_loader, f'epoch {i + 1} training...'):
            X = X.to('cuda')
            y = y.to('cuda')

            y_pred = model(X)
            loss = criterion(y_pred, y)

            optimizer.zero_grad()
            loss.backward()
            train_loss += loss.item()
            optimizer.step()

        model.eval()

        for X, y in tqdm(dev_loader, f'epoch {i + 1} evaluating...'):
            X = X.to('cuda')
            y = y.to('cuda')

            y_pred = model(X)
            loss = criterion(y_pred, y)
            dev_loss += loss.item()
        
        train_losses.append(train_loss / len(train_loader))
        dev_losses.append(dev_loss / len(dev_loader))

        if i % 5 == 4:
            torch.save(model, './model/model.pt')
            print(f'epoch {i + 1} model saved.')

    plt.plot(range(epoch), train_losses, label='train')
    plt.plot(range(epoch), dev_losses, label='dev')

    plt.legend()
    plt.show()

    full_pred = np.array([])
    full_real = np.array([])
    for X, y in tqdm(test_loader, 'Testing...'):
        X = X.to('cuda')
        y = y.to('cuda')

        y_pred = model(X).cpu().detach().numpy()
        y_pred = np.argmax(y_pred, axis=1)
        full_pred = np.concatenate([full_pred, y_pred])
        full_real = np.concatenate([full_real, y.cpu().numpy()])
    
    report = classification_report(full_real, full_pred, digits=3)
    mat = confusion_matrix(full_real, full_pred)
    print(report)
    print(mat)