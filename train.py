import torch
import torch.optim as optim
import torch.nn as nn
import torchvision
from torchvision.datasets import MNIST
from torchvision import transforms
from model import ConvNN
from utils import get_accuracy
import time
import copy

# At least one transformation
transform = transforms.Compose([
    transforms.ToTensor()
])

trainset = MNIST('MNIST', transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=2,
                                          shuffle=True, num_workers=2)

testset = MNIST('MNIST', train = False, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=2,
                                          shuffle=True, num_workers=2)

Debug_flag = 0
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

epoch = 10
model = ConvNN()
model = model.to(device)
lr = 1e-4
optimizer = optim.Adam(model.parameters(), lr=lr)
loss_fn = nn.CrossEntropyLoss()
exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size= 50, gamma= 0.1)

since = time.time()
best_model_wts = copy.deepcopy(model.state_dict())
best_acc = 0.0

for i in range(epoch):

    loss_show = 0
    for j, train_data in enumerate(trainloader):
        optimizer.zero_grad()
        if Debug_flag == 10:
            break
        train_img = train_data[0].to(device)
        label = train_data[1].to(device)

        pred = model(train_img)

        loss = loss_fn(pred, label)
        loss_show += loss
        loss.backward()
        optimizer.step()
        exp_lr_scheduler.step()
        Debug_flag += 1

    print(loss_show)
    print("-" * 20)
    Debug_flag = 0
    for j, test_data in enumerate(testloader):
        if Debug_flag == 2:
            break
        test_img = test_data[0].to(device)
        label = test_data[1].to(device)

        _, pred = model(test_img, apply_softmax=True).max(dim=1)
        print(pred)
        print(label)
        acc = get_accuracy(pred, label)
        print(acc)
        Debug_flag += 1


    print("{0:2d} epochs ends".format(i))

