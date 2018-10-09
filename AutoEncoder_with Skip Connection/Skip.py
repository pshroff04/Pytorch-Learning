
import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt
import numpy as np

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3,4,3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(4,8,3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(8,16,3, stride =2, padding=1)
        self.deconv1 = nn.ConvTranspose2d(16,8,4, stride =2, padding=1)
        self.deconv2 = nn.ConvTranspose2d(16,4,4, stride =2, padding=1)
        self.deconv3 = nn.ConvTranspose2d(8,3,4, stride =2, padding=1)

    def forward(self, x):
        x1 = F.relu(self.conv1(x))
        x2 = F.relu(self.conv2(x1))
        x3 = F.relu(self.conv3(x2))
        x4 = F.relu(self.deconv1(x3))
        x5 = F.relu(self.deconv2(torch.cat((x4,x2), dim=1)))   
        x6 = self.deconv3(torch.cat((x5,x1), dim=1))
        return x6



### Create an instance of the Net class
net = Net()

## Loading the training and test sets
# Converting the images for PILImage to tensor, so they can be accepted as the input to the network
transform = transforms.ToTensor()

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=5, shuffle=True)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=5, shuffle=False)

### Define the loss and create your optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(net.parameters(), lr = 0.001)

### Main training loop
for epoch in range(5):
    total_batch_loss = 0.0
    for i, data in enumerate(trainloader, 0):

        ## Getting the input and the target from the training set
        input, dummy = data
        target = input
        out = net(input)
        loss = criterion(out, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_batch_loss += loss.item()

    #print('For %d epoch, average loss is %.3f'%(epoch, total_batch_loss/10000))
        
        if (i+1)%5000 == 0 and i!=0 :
            print('For %d epoch, %5d minibatch, average loss is %.5f'%(epoch, (i+1), total_batch_loss/5000))
            total_batch_loss = 0.0


### Testing the network on 10,000 test images and computing the loss
testLoss = 0.0
with torch.no_grad():
    for data in testloader:
        input, dummy = data
        target = input
        out = net(target)
        loss = criterion(out, target)
        testLoss += loss.item()
    print('Test Set Loss %.6f' %(testLoss/2000))

### Displaying or saving the results as well as the ground truth images for the first five images in the test set

with torch.no_grad():
    testiterator = iter(testloader)
    o, _ = testiterator.next()
    testout = net(o)

#Display
plt.imshow(torchvision.utils.make_grid(testout).numpy().transpose((1,2,0)))
plt.show()

plt.imshow(torchvision.utils.make_grid(o).numpy().transpose((1,2,0)))
plt.show()

#save

torchvision.utils.save_image(torch.cat((testout,o)), "Skip.png", nrow=5)