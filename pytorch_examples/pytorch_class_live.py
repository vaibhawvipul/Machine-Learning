#author Vipul Vaibhaw

import torch 
import torch.nn as nn 
# Linear Layer in pytorch is equivalent fully connected layer or dense
import torch.nn.functional as F 
# 1st layer to be a nn.Conv2d(layer), F.conv2d(custom layer)
import torch.optim as optim 

#x = torch.tensor(([5.1,4.3,2.5],[5.1,4.3,2.5],[5.1,4.3,2.5]), 
#    dtype=torch.float) # 3X3 tensor

# size 4X3 

#x.size() # size of my tensor

# converting from numpy to pytorch tensor

#import numpy as np 

#a = np.array([1,2,3]) # numpy array
#b = torch.from_numpy(a)

#print(b)

#y = torch.rand(3,3) # initialized a random tensor of size 3X3
# take values from normal distribution of mean 0 and std dev 1.


#add_x_and_y = torch.add(x,y)

#print(add_x_and_y)

#x1 = torch.tensor(([5.1,4.3],[5.1,4.3],[5.1,4.3]), 
#    dtype=torch.float) # 2X3 tensor

#z1 = x1.view(-1,2)

#print(z1)
#print(z1.size())


import torchvision as tv 
import torchvision.transforms as transforms

dataset_transform = transforms.Compose([tv.transforms.ToTensor(),
            tv.transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
# Assignment - read about batch norm and 
# how it is different from normalization, 
# how to do it in pytorch! 

trainset = tv.datasets.CIFAR10(root="./data", 
            train=True, transform=dataset_transform, download=False)

dataloader = torch.utils.data.DataLoader(trainset,
            batch_size=4,num_workers=4)
        
class ConvModel(nn.Module):
    def __init__(self):
        super(ConvModel,self).__init__()

        self.conv1 = nn.Conv2d(3,6,5)
        self.pool = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(6,16,5)
        self.fc1 = nn.Linear(16*5*5,120)
        self.fc2 = nn.Linear(120,84)
        self.fc3 = nn.Linear(84,10)

    def forward(self,x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1,16*5*5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

net = ConvModel()
loss_func = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(),lr=0.001,
            momentum=0.9,nesterov=True,weight_decay=1e-6)

# training 
for epoch in range(2):
    running_loss = 0.0
    for i,data in enumerate(dataloader,0):

        inputs,labels = data 
        optimizer.zero_grad()

        #forward prop
        ouputs = net(inputs)
        loss = loss_func(ouputs,labels)

        #backward
        loss.backward() #autograd in pytorch
        optimizer.step()

        running_loss += loss.item()

        if i % 2000 == 1999:
            print("[epoch: %d, minibatch: %5d] loss: %.3f"%(epoch+1,i+1,running_loss))
            running_loss = 0.0

print("Yay!! Training Finished!")

