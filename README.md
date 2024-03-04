# Simple-MNIST-using-Pytorch
I have created a simple neural network for predicting the hand-written digits using the famous MNIST dataset. 
I have used Pytorch to implement the code.

#Firstly we start by importing all the dependencies
pip install torchvision
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

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
    
transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,),(0.5,))])
train_dataset=torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader=DataLoader(train_dataset,batch_size=64, shuffle=True)

model=SimpleNN()
Criteria=nn.CrossEntropyLoss()
optimizer=optim.SGD(model.parameters(), lr=0.01)

epochs=5
for epoch in range(epochs):
    for images, labels in train_loader:
        optimizer.zero_grad
        outputs=model(images)
        loss=Criteria(outputs, labels)
        optimizer.step()
        
    print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item()}')
          
torch.save(model.state_dict(),'mnist_model.pth')
