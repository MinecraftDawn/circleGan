import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import ToTensor
from torchvision.io import read_image
from torchsummary import summary
import os


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
# DEVICE = 'cpu'
EPOCHS = 300
print(f'Using {DEVICE}')

class VanGoghDataset(Dataset):
    def __init__(self, img_dir, transform=None, target_transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform
        self.img_path = os.listdir(self.img_dir)
        self.img_path += [''] * len(self.img_path)


    def __len__(self):
        return len(self.img_path)

    def __getitem__(self, index):
        img_path = self.img_path[index]
        if not img_path:
            image = torch.rand((100,100))
            label = 0
        else:
            # image = read_image(self.img_dir + img_path)
            image = torch.zeros((100,100)).float()
            label = 1

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        # sample = {'image': image, 'label': label}
        sample = (image, label)

        return sample

trd = VanGoghDataset('./images/Van_Gogh/')
dl = DataLoader(trd, batch_size=2000, shuffle=True)

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        # self.flatten = nn.Flatten()
        self.linear = nn.Sequential(
            nn.MaxPool2d(5,5),
            nn.Flatten(),
            nn.Linear(20 * 20, 10),
            nn.ReLU(),
            nn.Linear(10, 5),
            nn.ReLU(),
            # nn.Linear(1000, 100),
            # nn.ReLU(),
            nn.Linear(5, 2),
            nn.LogSoftmax()
        )

    def forward(self, x):
        # x = self.flatten(input)
        x = self.linear(x)
        return x

model = NeuralNetwork().to(DEVICE)
print(model)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(DEVICE), y.to(DEVICE)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


for t in range(EPOCHS):
    print(f"Epoch {t+1}\n-------------------------------")
    train(dl, model, loss_fn, optimizer)
print("Done!")