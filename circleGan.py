import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import ToTensor, Compose, CenterCrop, RandomCrop, RandomHorizontalFlip, RandomSizedCrop, \
    Normalize
from torchvision.io import read_image
from torchsummary import summary
import os
import matplotlib.pyplot as plt
import numpy as np

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
# DEVICE = 'cpu'
EPOCHS = 1
BATCH_SIZE = 10
print(f'Using {DEVICE}')


class VanGoghDataset(Dataset):
    def __init__(self, img_dir, transform=None, target_transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform
        self.img_path = os.listdir(self.img_dir)

    def __len__(self):
        return len(self.img_path)

    def __getitem__(self, index):
        img_path = self.img_path[index]
        image = read_image(self.img_dir + img_path)

        image = image.float()
        image = (image - 0.5) / 255

        label = torch.ones(1)

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        # sample = {'image': image, 'label': label}
        sample = (image, label)

        return sample


trans = Compose([RandomSizedCrop(40), RandomHorizontalFlip(0.5)])

trd = VanGoghDataset('./images/Van_Gogh/',
                     transform=trans)
dl = DataLoader(trd, batch_size=BATCH_SIZE, shuffle=True)


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            # nn.Flatten(),
            # nn.Linear(3*80*80, 5000),
            # nn.LeakyReLU(0.2),
            # nn.Linear(5000, 1000),
            # nn.LeakyReLU(0.2),
            # nn.Linear(1000, 100),
            # nn.LeakyReLU(0.2),
            # nn.Linear(100, 1),
            # nn.Sigmoid()

            nn.Flatten(),
            nn.Linear(3*40*40, 1000),
            nn.Dropout(0.5),
            nn.LeakyReLU(0.2),
            nn.Linear(1000, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.main(x)

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(128, 512),
            nn.ReLU(),
            nn.Linear(512, 2048),
            nn.ReLU(),
            nn.Linear(2048, 40*40*3),
            nn.Tanh()
        )

    def forward(self, x):
        return self.main(x)

def d_loss_func(inputs, targets):
    return nn.BCELoss()(inputs, targets)

def g_loss_func(inputs):
    target = torch.ones(inputs.shape[0], 1).to(DEVICE)
    return nn.BCELoss()(inputs, target)

def showImg(img:torch.Tensor):
    img = np.rollaxis(img.numpy(), 0, 3)
    img = (img + 0.5) * 255
    img = img.astype(np.uint8)
    plt.imshow(img)
    plt.show()


discriminator = Discriminator().to(DEVICE)
generator = Generator().to(DEVICE)

print(discriminator)
print(generator)

d_optimizer = Adam(discriminator.parameters(), lr=0.0001, betas=(0.5, 0.999))
g_optimizer = Adam(generator.parameters(), lr=0.1, betas=(0.5, 0.999))


def train(dl, discriminator, generator, d_optimizer, g_optimizer):
    size = len(dl.dataset)
    for batch, (X, _) in enumerate(dl):
        # Discriminator
        batchSize = X.shape[0]
        realInput = X.to(DEVICE)
        realOutput = discriminator(realInput)
        realLabel = torch.ones(batchSize, 1).to(DEVICE)

        noise = torch.rand(batchSize, 128).to(DEVICE)
        fakeInput = generator(noise)
        fakeOutput = discriminator(fakeInput)
        fakeLabel = torch.zeros(batchSize, 1).to(DEVICE)

        outputs = torch.cat((realOutput, fakeOutput), 0)
        label = torch.cat((realLabel, fakeLabel), 0)

        # Discriminator Backpropagation
        d_optimizer.zero_grad()
        d_loss = d_loss_func(outputs, label)
        d_loss.backward()
        d_optimizer.step()

        for _ in range(100):
            # Generator
            noise = torch.rand(batchSize, 128).to(DEVICE)
            fakeInput = generator(noise)
            fakeOutput = discriminator(fakeInput)

            # Generator Backpropagation
            g_loss = g_loss_func(fakeOutput)
            g_optimizer.zero_grad()
            g_loss.backward()
            g_optimizer.step()

        if batch % 100 == 0:
            current = batch * len(X)
            print(f"d_loss: {d_loss.item():>7f} g_loss: {g_loss.item():>7f} [{current:>5d}/{size:>5d}]")

for t in range(EPOCHS):
    print(f"Epoch {t+1}\n-------------------------------")
    train(dl, discriminator, generator, d_optimizer, g_optimizer)
print("Done!")


# model = Discriminator().to(DEVICE)
# print(model)
#
# loss_fn = nn.BCELoss()
# optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
#
#
# def train(dataloader, model, loss_fn, optimizer):
#     size = len(dataloader.dataset)
#     for batch, (X, y) in enumerate(dataloader):
#         X, y = X.to(DEVICE), y.to(DEVICE)
#
#         # Compute prediction error
#         pred = model(X)
#         loss = loss_fn(pred, y)
#         # print(pred)
#         # print(y)
#         # print(loss)
#
#         # Backpropagation
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#
#         if batch % 100 == 0:
#             loss, current = loss.item(), batch * len(X)
#             print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
#
# for t in range(EPOCHS):
#     print(f"Epoch {t+1}\n-------------------------------")
#     train(dl, model, loss_fn, optimizer)
# print("Done!")