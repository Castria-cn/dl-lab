import os
import cv2
import torch
import numpy as np
from torch import nn
from tqdm import tqdm
from torchvision import transforms
from tensorboardX import SummaryWriter
from torchvision.models import AlexNet
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader, Dataset, random_split

class MyDataset(Dataset):
    def __init__(self, path):
        classes = os.listdir(path)
        self.x = []
        self.y = []
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        for y, cls in enumerate(tqdm(classes, '读取文件中...')):
            images = os.listdir(f'{path}/{cls}/')
            for img in images:
                img = cv2.imread(f'{path}/{cls}/{img}')
                self.x.append(self.transform(img))
                self.y.append(y)
    
    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        return self.x[index], self.y[index]

class AlexxNet(nn.Module):
    def __init__(self, output_dim):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, output_dim),
        )
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
    
epoch = 50
lr = 1e-3
batch_size = 16
load = None # './model/model.pt'
    
if __name__ == '__main__':
    writer = SummaryWriter()
    dataset = MyDataset('./caltech-101/101_ObjectCategories')
    
    train, test = random_split(dataset, [.9, .1], generator=torch.Generator().manual_seed(42))
    train, dev = random_split(train, [.8, .2])

    train_loader = DataLoader(train, batch_size=batch_size)
    dev_loader = DataLoader(dev, batch_size=batch_size)
    test_loader = DataLoader(dev, batch_size=batch_size)

    model = AlexNet(101)
    model.to('cuda')

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    train_cnt, dev_cnt = 0, 0

    for i in range(epoch):
        model.train()
        for X, y in tqdm(train_loader, f'epoch {i + 1} training...'):
            X = X.to('cuda')
            y = y.to('cuda')

            y_pred = model(X)
            loss = criterion(y_pred, y)

            optimizer.zero_grad()
            loss.backward()

            writer.add_scalar('Train loss', loss.item(), train_cnt)
            train_cnt += 1
            optimizer.step()

        model.eval()

        for X, y in tqdm(dev_loader, f'epoch {i + 1} evaluating...'):
            X = X.to('cuda')
            y = y.to('cuda')

            y_pred = model(X)
            loss = criterion(y_pred, y)
            writer.add_scalar('Dev loss', loss.item(), dev_cnt)
            dev_cnt += 1

        if i % 5 == 4:
            torch.save(model, './model/model.pt')
            print(f'epoch {i + 1} model saved.')
    
    full_pred = np.array([])
    full_real = np.array([])

    for X, y in tqdm(test_loader, '测试中...'):
        X = X.to('cuda')
        y = y.to('cuda')

        y_pred = model(X).cpu().detach().numpy()
        y_pred = np.argmax(y_pred, axis=1)
        full_pred = np.concatenate([full_pred, y_pred])
        full_real = np.concatenate([full_real, y.cpu().numpy()])
    
    report = classification_report(full_real, full_pred, digits=3)

    print(report)
