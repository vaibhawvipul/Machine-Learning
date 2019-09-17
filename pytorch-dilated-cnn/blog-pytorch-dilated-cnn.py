import torch
import torchvision as tv
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Loading and Transforming data
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.4914, 0.4822, 0.4466), (0.247, 0.243, 0.261))])
trainTransform  = tv.transforms.Compose([tv.transforms.ToTensor(), tv.transforms.Normalize((0.4914, 0.4822, 0.4466), (0.247, 0.243, 0.261))])
trainset = tv.datasets.CIFAR10(root='../data', train=True,
                                        download=True, transform=transform)
dataloader = torch.utils.data.DataLoader(trainset, batch_size=1, shuffle=False, num_workers=4)

# Writing our model
class DilatedCNN(nn.Module):
    def __init__(self):
        super(DilatedCNN,self).__init__()
        self.convlayers = nn.Sequential(
            nn.Conv2d(in_channels = 3, out_channels = 6, kernel_size = 5, stride = 1, padding = 0, dilation=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size = 5, stride = 1, padding= 0, dilation = 2), 
            nn.ReLU(),
        )
        self.fclayers = nn.Sequential(
            nn.Linear(4096,120),
            nn.ReLU(),
            nn.Linear(120,84),
            nn.ReLU(),
            nn.Linear(84,10)
        )
    
    def forward(self,x):
        x = self.convlayers(x)
        x = x.view(-1,4096)
        x = self.fclayers(x)
        return x


net = DilatedCNN()

#optimization and score function
loss_function = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(),lr=0.001,momentum=0.5)

#training of the model

for epoch in range(2):
	running_loss= 0.0
	for i,data in enumerate(dataloader,0):
		inputs, labels = data

		optimizer.zero_grad()

		outputs = net(inputs)
		loss = loss_function(outputs, labels)
		loss.backward()
		optimizer.step()

		running_loss += loss.item()
		if i%2000==1999:
			print('[%d, %5d] loss: %.3f'%(epoch+1,i+1,running_loss/2000))
			running_loss = 0.0

print("Training finshed! Yay!!")

