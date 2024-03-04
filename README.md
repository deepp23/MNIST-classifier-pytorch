

# MNIST-classifier-pytorch
This project demonstrates a simple neural network implementation for handwritten digit recognition using the MNIST dataset and the PyTorch framework


## Deployment

1. Dependencies

```bash
pip install torchvision
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
```

2. Neural Network Model

```bash
class SimpleNN(nn.Module):
  def __init__(self):
    super(SimpleNN,self).__init__()
    self.fc1=nn.Linear(28*28,128)  
    self.relu=nn.ReLU()             
    self.fc2=nn.Linear(128,10)       
     
  def forward(self,x):
    x=x.view(-1,28*28)              
    x=self.fc1(x)
    x=self.relu(x)
    x=self.fc2(x)
    return x   
```
3. Data Preparation
```bash
transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,),(0.5,))])
train_dataset=torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader=DataLoader(train_dataset,batch_size=64, shuffle=True)
```
4. Training Loop
```bash
model=SimpleNN()
Criteria=nn.CrossEntropyLoss()
optimizer=optim.SGD(model.parameters(), lr=0.01)

epochs=5
for epoch in range(epochs):
  for images, labels in train_loader:
    optimizer.zero_grad()   
    outputs=model(images)
    loss=Criteria(outputs, labels) 
    loss.backward()         
    optimizer.step()        
     
  print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item()}')
torch.save(model.state_dict(),'mnist_model.pth')

```
5.Test Evalution
```bash
test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
correct = 0
total = 0
with torch.no_grad(): 
for images, labels in test_loader:
    outputs = model(images)
     _, predicted = torch.max(outputs.data, 1)  
    total += labels.size(0)
    correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f'Test Accuracy: {accuracy}%')

```
## Related

Here are some related projects

[MNIST-CNN](https://github.com/AmritK10/MNIST-CNN/blob/master/README.md)

[MNIST-TensorFlow](https://github.com/golbin/TensorFlow-MNIST/blob/master/README.md)
