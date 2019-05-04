# author - Vipul Vaibhaw
# Thanks for coming here guys, do checkout my blog
# https://vipulvaibhaw.com/

# This code will be a self explanatory code with a lot of comments.
# However, I am still going to write a blog on this to help more 
# poeple out there :) 

# firstly we will import the necessary modules
import torch
import torch.nn as nn # torch.nn contains all the necessary tools we would need to code a neural network.
import torch.nn.functional as F # contains activation functions we need
import torch.optim as optim  # as the name suggests, it has optimizers

# Now we will import the modules which will allow us to get cifar-10 dataset in our code
import torchvision as tv 
import torchvision.transforms as transforms # this will allow us to do various transforms on our images like Normalization.

# Great, Let's import and prepare our dataset

# Writing our transformation pipeline
transform = transforms.Compose([tv.transforms.ToTensor(),
			tv.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))]) # check out the blog for explanation of these numbers

# Download the dataset and apply the above transform to it
trainset = tv.datasets.CIFAR10(root="./data",train=True,download=False,transform=transform) # set download flag as true if you are running this code for the first time

# Load the data in the memory, a suggested read - https://stackoverflow.com/questions/53332663/torchvision-0-2-1-transforms-normalize-does-not-work-as-expected/53334458#53334458
dataloader = torch.utils.data.DataLoader(trainset,batch_size=4, shuffle=False, num_workers=4)

# Awesome, now it is time to write some models!

class OurModel(nn.Module):

    def __init__(self):
        super(OurModel,self).__init__()
        self.conv1 = nn.Conv2d(3,6,5) #check out the blog for numbers! and difference between nn.Conv2d and F.conv2d
        self.pool = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(6,16,5)
        self.fc1 = nn.Linear(16*5*5,120)
        self.fc2 = nn.Linear(120,84)
        self.fc3 = nn.Linear(84,10)

    def forward(self,x):
		x = self.pool(F.relu(self.conv1(x))) 
		x = self.pool(F.relu(self.conv2(x)))
		x = x.view(-1,16*5*5) # flatten out the cube so that it can be passed into Fully connected layer
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		x = self.fc3(x)
		return x

# We have defined the model now, let us define loss function and optimizer

net = OurModel()
loss_func = nn.CrossEntropyLoss() 
optimizer = optim.SGD(net.parameters(),lr=0.001,momentum=0.5)

# time to train
for epoch in range(2):
    running_loss= 0.0
    for i,data in enumerate(dataloader,0):
        inputs, labels = data
        optimizer.zero_grad()
        # forward prop
        outputs = net(inputs)
        loss = loss_func(outputs, labels)
        # backprop
        loss.backward() # compute gradients
        optimizer.step() # update parameters
        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print('[epoch: %d, minibatch: %5d] loss: %.3f' %(epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0
        
print("Training finshed!")
