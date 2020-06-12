# -*- coding: utf-8 -*-
import argparse

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import CifarNet as cfn
from DenseNet import DenseNet as densnet
from ResNet import resnet50

parser = argparse.ArgumentParser()
parser.add_argument('--test', action='store_true')
parser.add_argument('--show_data', action='store_true')

# global data
classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
DATA_PATH = './data'
MODEL_PATH = './resnet50_mb4_lr0.1.pth'
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def main():

    args = parser.parse_args()
    if args.test:
        test_dataloader = cifar10_dataloader(train=False,
            batch_size=4, shuffle=True, num_workers=2)
        # model = cfn._cifarnet(pretrained=args.test, path=MODEL_PATH).to(device)
        # model = densnet(growth_rate=32, num_classes=10).to(device)
        model = resnet50(num_classes=10).to(device)
        state_dict = torch.load(MODEL_PATH)
        model.load_state_dict(state_dict)
        test(test_dataloader, model, args.show_data)
        return

    if args.show_data:
        dataloader = cifar10_dataloader(train=False,
            batch_size=4, shuffle=True, num_workers=0)
        show_data(dataloader)
        return

    train_dataloader = cifar10_dataloader(train=True, 
        batch_size=4, shuffle=True, num_workers=2)
    val_dataloader = cifar10_dataloader(train=False,
        batch_size=4, shuffle=True, num_workers=0)
    
    # model = cfn._cifarnet().to(device)
    # model = densnet(growth_rate=32, num_classes=10).to(device)
    model = resnet50(num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=0.0001)

    best_loss = 10.0
    for epoch in range(10):
        epoch_loss = train(train_dataloader, model, criterion, optimizer, epoch)
        print('[train][%depoch] loss: %.5f'%(epoch, epoch_loss))
        val(val_dataloader, model, criterion, epoch)
        if best_loss > epoch_loss:
            best_loss = epoch_loss
            torch.save(model.state_dict(), MODEL_PATH)

def cifar10_dataloader(root=DATA_PATH, train=True, transform=None, 
	shuffle=False, download=True, batch_size=4, num_workers=0):

	if transform is None:
		transform = transforms.Compose([
			transforms.ToTensor(),
			transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
			])

	dataset = torchvision.datasets.CIFAR10(root=root, 
		train=train, transform=transform, download=download)
	
	dataloader = torch.utils.data.DataLoader(dataset, 
		batch_size=batch_size, num_workers=num_workers, shuffle=shuffle)

	return dataloader 

def train(dataloader, model, criterion, optimizer, epoch):
    running_loss = 0.0
    total_loss = 0.0
    for i, data in enumerate(dataloader):
        images, labels = data[0].to(device), data[1].to(device)
        logit = model(images)
        loss = criterion(logit, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        total_loss += loss.item()
        # 2000개 minibatch 마다 loss 평균 프린트
        if i%2000 == 1999:
            print('epoch[%d]iter[%d] loss: %.5f'%(epoch, i, running_loss/2000.0))
            running_loss = 0.0
    return total_loss/len(dataloader)
def val(dataloader, model, criterion, epoch):
    loss = 0
    with torch.no_grad():
        for data in dataloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = model(images)
            loss += criterion(outputs, labels).item()
    print('[val][%depoch] loss:%.5f'%(epoch, loss/len(dataloader)))

def test(dataloader, model, show_data):   
    if show_data:
        dataiter = iter(dataloader)
        images, labels = dataiter.next()
        print(images.size())
        # show images
        imshow(torchvision.utils.make_grid(images))  
        output = model(images.to(device))
        _, predicted = torch.max(output, 1)
        print('GT', ' '.join('%6s' % classes[labels[j]] for j in range(4)))
        print('PT', ' '.join('%6s' % classes[predicted[j]] for j in range(4)))
        print()

    correct = 0
    total = 0
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))
    with torch.no_grad():
        for data in dataloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            c = (predicted == labels).squeeze()
            for i in range(4):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1

    print('Accuracy of the network on the 10000 test images: %d %%' % (
        100 * correct / total))
    print()
    for i in range(10):
        print('Accuracy of %5s : %2d %%' % (
            classes[i], 100 * class_correct[i] / class_total[i]))

def show_data(dataloader):
    dataiter = iter(dataloader)
    images, labels = dataiter.next()

    # show images
    imshow(torchvision.utils.make_grid(images))

    # print labels
    print(' '.join('%10s' % classes[labels[j]] for j in range(4)))

def imshow(img):
    import matplotlib.pyplot as plt
    import numpy as np    

    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

if __name__ == '__main__':
    main()
