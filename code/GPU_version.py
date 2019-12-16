#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  3 19:36:34 2019

@author: ziyushu
"""
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import scipy.sparse.linalg
import datetime
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
torch.manual_seed(1237)
dtype = torch.cuda.FloatTensor
class GTSBDataset(torch.utils.data.Dataset):
    def __init__(self,root,train,transform = False):
        if train:
            data = np.load(root+'data.npy')
            label = np.load(root+'label.npy')
        else:
            data = np.load(root+'test_data.npy')
            label = np.load(root+'test_label.npy')
        
        if transform:
            Pdata = []
            for i in range(data.shape[0]):
                tmp = data[i]
                tmp = np.transpose(tmp,(2,1,0))
                tmp = Image.fromarray(tmp.astype('uint8'), 'RGB')
                tmp = transform(tmp)
                Pdata.append(tmp)
            data = torch.stack(Pdata,dim=0)
            self.x_data = data
        else:
            self.x_data = torch.from_numpy(data)
        self.len = data.shape[0]  
        #label = label.astype(float32)
        self.y_data = torch.from_numpy(label).long()
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]
    
    def __len__(self):
        return self.len
    


class MySpec_conv(torch.autograd.Function):
    @staticmethod
    def forward(self, input,weight):
        self.save_for_backward(input,weight)
        return torch.tensor(0.0).cuda()

    @staticmethod
    def backward(self, grad_output):
        input,weight = self.saved_tensors
        xr = input.clone().detach()
        xi = torch.cuda.FloatTensor(xr.shape[:]).fill_(0)
        xri = torch.stack((xr,xi),4)
        Fxri = torch.fft(xri,2)
        absFxri = torch.sum(torch.mul(Fxri,Fxri),4)
        absFxri = torch.sqrt(absFxri)
        
        m1,_ = torch.max(absFxri,2)
        m1,_ = torch.max(m1,2)
        m1 = torch.unsqueeze(m1,2)
        m1 = torch.unsqueeze(m1,2)
        absFxri = torch.div(absFxri,m1)
        absFxri = torch.mul(absFxri,absFxri)
        absFxri = torch.mul(absFxri,absFxri)
        
        absFxri = torch.mul(absFxri,absFxri)
        absFxri = torch.mul(absFxri,absFxri)
        absFxri = torch.mul(absFxri,absFxri)

        
        absFxri = torch.unsqueeze(absFxri,4)  
        Fxri = torch.ifft(torch.mul(Fxri,absFxri),2)
        Fxri = Fxri[:,:,:,:,0]
        
        
        del m1
        del xr
        del xi
        del xri
        del absFxri
        Fxri = weight*Fxri
        return Fxri,None
'''    
class MySpec_fc(torch.autograd.Function):
    @staticmethod
    def forward(self, input,weight):
        self.save_for_backward(input,weight)
        return torch.tensor(0.0).cuda()

    @staticmethod
    def backward(self, grad_output):
        input,weight = self.saved_tensors
        inputc = input.clone()
        weightc = weight.clone()
        weightc = weightc.detach().cpu().numpy()
        x = inputc.detach().cpu().numpy()
        s,v,d = scipy.sparse.linalg.svds(x,k=1)
        #u1 = x.dot(d.T)
        #v1 = (x.T).dot(u1)
        #coe = np.linalg.norm(u1)/np.linalg.norm(v1)
        diff_x = v[0]*(s.dot(d))*weightc
        return torch.from_numpy(diff_x).cuda(),None  
'''   
class OldSpec_conv(torch.autograd.Function):
    @staticmethod
    def forward(self, input,weight):
        self.save_for_backward(input,weight)
        return torch.tensor(0.0).cuda()

    @staticmethod
    def backward(self, grad_output):
        input,weight = self.saved_tensors
        x = input.clone().detach()

        [Io,Ii,h,w] = x.shape
        x = torch.reshape(x,[Io,-1])
        v = torch.randn(Ii*h*w,1).cuda()
        u = torch.matmul(x,v)
        v = torch.matmul(torch.transpose(x,0,1),u)
        weight = weight*torch.sum(torch.mul(u,u))/torch.sum(torch.mul(v,v))
        x = torch.matmul(u,torch.transpose(v,0,1))*weight
        x = torch.reshape(x,[Io,Ii,h,w])
        return x,None
'''    
class OldSpec_fc(torch.autograd.Function):
    @staticmethod
    def forward(self, input,weight):
        self.save_for_backward(input,weight)
        return torch.tensor(0.0).cuda()

    @staticmethod
    def backward(self, grad_output):
        input,weight = self.saved_tensors
        inputc = input.clone()
        weightc = weight.clone()
        weightc = weightc.detach().cpu().numpy()
        x = inputc.detach().cpu().numpy()
        [h,w] = x.shape
        v = np.random.normal(size = [w,1])
        u = x.dot(v)
        v = x.T.dot(u)
        coe = np.linalg.norm(u)/np.linalg.norm(v)
        diff_x = coe*(u.dot(v.T))*weightc
        return torch.from_numpy(diff_x).cuda(),None
'''


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3)
        self.conv1_bn = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, 3)
        self.conv2_bn = nn.BatchNorm2d(32)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(32, 64, 3)
        self.conv3_bn = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 64, 3)
        self.conv4_bn = nn.BatchNorm2d(64)
        #self.pool2 = nn.MaxPool2d(2, 2)
        self.conv5 = nn.Conv2d(64, 128, 3)
        self.conv5_bn = nn.BatchNorm2d(128)
        self.conv6 = nn.Conv2d(128, 128, 3)
        self.conv6_bn = nn.BatchNorm2d(128)
       
        #self.pool5 = nn.MaxPool2d(2, 2)
        self.avgpool = nn.AvgPool2d(kernel_size=1, stride=1, padding=0)
        
        
        
        self.fc1 = nn.Linear(2048, 512)
        self.fc2 = nn.Linear(512, 43)
        self.drop_layer2 = nn.Dropout(p=0.5)
        self.drop_layer1 = nn.Dropout(p=0.0)
    def forward(self, x):
        #x[4,3,32,32]
        x = F.pad(x,pad=[1,1,1,1])#循环卷积需要循环pad
        x = F.relu(self.conv1_bn(self.conv1(x)))
        x = F.pad(x,pad=[1,1,1,1])#循环卷积需要循环pad
        x = self.pool(F.relu(self.conv2_bn(self.conv2(x))))    
        x = self.drop_layer1(x)
        #x[4,32,16,16]
        x = F.pad(x,pad=[1,1,1,1])#循环卷积需要循环pad
        x = F.relu(self.conv3_bn(self.conv3(x)))
        x = F.pad(x,pad=[1,1,1,1])#循环卷积需要循环pad
        x = self.pool(F.relu(self.conv4_bn(self.conv4(x))))
        x = self.drop_layer1(x)
        #x[4,64,8,8]
        x = F.pad(x,pad=[1,1,1,1])#循环卷积需要循环pad
        x = F.relu(self.conv5_bn(self.conv5(x)))
        x = F.pad(x,pad=[1,1,1,1])#循环卷积需要循环pad
        x = F.relu(self.conv6_bn(self.conv6(x)))   
        x = self.pool(x)
        x = self.drop_layer1(x) 
        
        #print(x.shape)
        #x[4,128,4,4]
        x = x.view(-1, 2048)
        #x[4,512*2*2]
        x = self.fc1(x)
        x = self.drop_layer2(x)
        x = self.fc2(x)

        return x,self.conv1.weight,self.conv2.weight,self.conv3.weight,self.conv4.weight,self.conv5.weight,self.conv6.weight,
     
def calculate_spec_fc(input):
    x = input.clone()
    x = x.detach().numpy()
    s,v,d = scipy.sparse.linalg.svds(x,k=1)
    return abs(v[0])
    
def calculate_spec_conv(input):
    inputc = input.clone()
    x = inputc.detach().numpy()
    [r,c,d,n] = x.shape
    fft2x = np.fft.fft2(x,axes = [2,3])
    absfft2x = np.abs(fft2x)
    output = 0.0
    for i in range(r):
        for j in range(c):
            tmp = np.amax(absfft2x[i,j,:,:])
            output += tmp              
    return tmp  

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=5),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5)),
])
trainset = GTSBDataset(root='./', train=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=32,shuffle=True, num_workers=0)
testset = GTSBDataset(root='./', train=False, transform=transform_test)
testloader = torch.utils.data.DataLoader(trainset, batch_size=100,shuffle=False, num_workers=0)

  

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

net = Net()
net.to(device)




criterion = nn.CrossEntropyLoss()
mySpec_conv = MySpec_conv.apply
oldSpec_conv = OldSpec_conv.apply
#mySpec_fc = MySpec_fc.apply
#oldSpec_fc = OldSpec_fc.apply
testacc = []
trainacc = []

optimizer = optim.SGD(net.parameters(), lr=1e-2, momentum=0.9,weight_decay=0)
#optimizer = optim.Adam(net.parameters(), lr=1e-3)

for epoch in range(200):  # loop over the dataset multiple times
    start = datetime.datetime.now()
    running_loss = 0.0
    num = 0
    total = 0
    correct = 0
    for i, data in enumerate(trainloader):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data[0].to(device), data[1].to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs,c1,c2,c3,c4,c5,c6= net(inputs)
        loss = criterion(outputs, labels)
        
        #Our Spec Nrom
        C_mySpec_conv = torch.tensor(0.1).to(device)
        #C_mySpec_fc = torch.tensor(0.01).to(device)
        loss_my_conv =  mySpec_conv(c1,C_mySpec_conv) + mySpec_conv(c2,C_mySpec_conv) + mySpec_conv(c3,C_mySpec_conv) + mySpec_conv(c4,C_mySpec_conv) + mySpec_conv(c5,C_mySpec_conv) + mySpec_conv(c6,C_mySpec_conv)
        #loss_my_fc = mySpec_fc(f1,C_mySpec_fc) + mySpec_fc(f2,C_mySpec_fc) + mySpec_fc(f3,C_mySpec_fc) + mySpec_fc(f4,C_mySpec_fc)
        
        
        #old spec nrom
        C_oldSpec_conv = torch.tensor(0.1).to(device)
        #C_oldSpec_fc = torch.tensor(0.01).to(device)
        loss_old_conv =  oldSpec_conv(c1,C_oldSpec_conv) + oldSpec_conv(c2,C_oldSpec_conv) + oldSpec_conv(c3,C_oldSpec_conv) + oldSpec_conv(c4,C_oldSpec_conv) + oldSpec_conv(c5,C_oldSpec_conv) + oldSpec_conv(c6,C_oldSpec_conv)
        #loss_old_fc = oldSpec_fc(f1,C_mySpec_fc) + oldSpec_fc(f2,C_mySpec_fc) + oldSpec_fc(f3,C_mySpec_fc) + oldSpec_fc(f4,C_mySpec_fc)
        
        loss = loss + loss_old_conv# + loss_old_conv#+ loss_my_conv 
        loss.backward()
        optimizer.step()

        num += 1
        total += labels.size(0)
        _, predicted = torch.max(outputs.data, 1)
        correct += (predicted == labels).sum().item()
        running_loss += loss.item()

    end = datetime.datetime.now()
    Trainacc = correct/total
    print((epoch + 1, running_loss / (i+1)),'trainacc',Trainacc,'time:',end-start)

    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs,_,_,_,_,_,_= net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        trainacc.append(Trainacc)
        testacc.append(correct / total)
        print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))            


            
            
            
            
print('Finished Training')




PATH = './GTSB_old_0.1.pth'
torch.save(net.state_dict(), PATH)

net = Net()
net.load_state_dict(torch.load(PATH))


correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs,_,_,_,_,_,_= net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the 10000 test images: %d %%' % (
    100 * correct / total))



correct = 0
total = 0
with torch.no_grad():
    for data in trainloader:
        images, labels = data
        outputs,_,_,_,_,_,_= net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the 10000 test images: %d %%' % (
    100 * correct / total))









