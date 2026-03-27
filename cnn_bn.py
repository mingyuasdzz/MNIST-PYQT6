import torch
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import datasets
import matplotlib.pyplot as plt
# lenet5

train_dataset = datasets.MNIST(root='./dataset/mnist',
                                train=True, 
                                transform=transforms.ToTensor(), 
                                download=True)
test_dataset = datasets.MNIST(root='./dataset/mnist',
                                train=False, 
                                transform=transforms.ToTensor(), 
                                download=True)

train_loader = DataLoader(dataset=train_dataset, 
                          batch_size=32, 
                          shuffle=True, 
                          num_workers=4)

test_loader = DataLoader(dataset=test_dataset, 
                          batch_size=1000, 
                          shuffle=False, 
                          num_workers=4)


class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 32, kernel_size=3, padding=1)  # 1*28*28 - > 32*28*28 - > 32*14*14
        self.conv2 = torch.nn.Conv2d(32, 64, kernel_size=3, padding=1)  # 32*14*14 -> 64*14*14 - > 64*7*7
        self.bn1 = torch.nn.BatchNorm2d(32)
        self.bn2 = torch.nn.BatchNorm2d(64)
        self.pooling = torch.nn.MaxPool2d(2, 2)
        self.relu = torch.nn.ReLU()
        self.fc1 = torch.nn.Linear(64*7*7, 128) 
        self.bn3 = torch.nn.BatchNorm1d(128)
        self.fc2 = torch.nn.Linear(128, 10)
        #self.dropout = torch.nn.Dropout(0.1)

    def forward(self, x):
        batch_size = x.size(0)
        x = self.pooling(self.relu(self.bn1(self.conv1(x))))
        x = self.pooling(self.relu(self.bn2(self.conv2(x))))
        x = x.view(batch_size, -1) # flatten
        x = self.relu(self.bn3(self.fc1(x)))
        x = self.fc2(x)
        return x
    
model = Net()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
criterion = torch.nn.CrossEntropyLoss(reduction='mean')
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
def train(epochs):
    #acc = []
    epoch_ls = []
    for epoch in range(epochs):
        epoch_ls.append(epoch)
        running_loss = 0.0
        correct = 0
        total = 0
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            

        epoch_acc = 100 * correct / total
        print(f'Epoch [{epoch+1}], Loss: {running_loss:.4f}, Accuracy: {epoch_acc:.2f}%')
    #plt.plot(epoch_ls, acc)
    #plt.xlabel('Epoch')
    #plt.ylabel('Accuracy')
    #plt.title('Training Accuracy')
    #plt.show()

def test():
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f'Accuracy on test set: {100 * correct / total:.2f}%')




def save_model(path='model.pth'):
    torch.save(model.state_dict(), path)
    print(f'Model saved to {path}')

def load_model(path='model.pth'):
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    model.eval()
    print(f'Model loaded from {path}')

    
if __name__ == '__main__':
    train(epochs=10)
    test()
    save_model()