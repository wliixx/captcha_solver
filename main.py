import torch
import torchvision
from torch import nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

torch.manual_seed(42)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print(f"устройство: {device}")

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
def train_one_epoch(model,loader, loss_fn, optimizer):
    model.train
    total_loss = 0.0
    correct = 0 
    total = 0
    
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = loss_fn(logits, labels)
        loss.backward()
        optimizer.step()
        batch_size = labels.size(0)
        total_loss += loss.item()*batch_size
        predictions = logits.argmax(dim=1)
        correct += (predictions == labels).sum().item()
        total += batch_size
        
    average_loss = total_loss / total
    accuracy = correct / total
    return average_loss, accuracy

def evaluate(model, loader, loss_fn):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    with torch.inference_mode():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            loss = loss_fn(logits, labels)

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size

            predictions = logits.argmax(dim=1)
            correct += (predictions == labels).sum().item()
            total += batch_size
        average_loss = total_loss / total
        accuracy = correct / total
        return average_loss, accuracy

def main():
    model = DigitClassifier().to(device)
    loss_fn = nn.CrossEntropyLoss()
    
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.001,
    )
    
    epochs = 5
    for epoch in range(epochs):
        train_loss, train_accuracy = train_one_epoch(
            model,
            train_loader,
            loss_fn,
            optimizer,
        )
        test_loss, test_accuracy = evaluate(
            model,
            test_loader,
            loss_fn
        )
        print(
            f"Эпоха {epoch + 1}/{epochs} | "
            f"Потери: {train_loss:.4f} | "
            f"Точность обучения: {train_accuracy:.2%} | "
            f"Точность теста: {test_accuracy:.2%}"
        )
        
    torch.save(
        model.state_dict(),
        'mnist_digit_model.pth',
    )
    
    print('модель сохранена')
    
    image, true_label = test_data[0]
    
    model.eval()
    with torch.inference_mode():
        image_batch = image.unsqueeze(0).to(device)
        logits = model(image_batch)
        prediction = logits.argmax(dim=1).item()
        
    print(f"Правильная цифра: {true_label}")
    print(f"Предсказание модели: {prediction}")

if __name__ == '__main__':
    main()
