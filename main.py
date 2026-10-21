import torch
import torchvision
from torch import nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

transform = transforms.ToTensor()

train_data = datasets.MNIST(
    root='data',
    train=True,
    download=True,
    transform=transform,
)
test_data = datasets.MNIST(
    root='data',
    train=False,
    download=True,
    transform=transform,
)

image, label = train_data[0]

train_loader = DataLoader(
    train_data,
    batch_size=64,
    shuffle=True,
)
test_loader = DataLoader(
    train_data,
    batch_size=64,
    shuffle=False,
)

class DigitClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.layers = nn.Sequential(
            nn.Linear(28 * 28, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )
    def forward(self, x):
        x = self.flatten(x)
        x = self.layers(x)
        return x
    
model = DigitClassifier()

loss_fn = nn.CrossEntropyLoss
loss = loss_fn(logits, label)
loss.backward()
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001,
)