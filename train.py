import torch
import torch.optim as optim
import torch.nn as nn
import torchvision
from torchvision.datasets import MNIST
from torchvision import transforms
from model import ConvNN

# At least one transformation
transform = transforms.Compose([
    transforms.ToTensor()
])

trainset = MNIST('MNIST', transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=512,
                                          shuffle=True, num_workers=2)

Debug_flag = 0
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

epoch = 160
model = ConvNN()
model = model.to(device)
lr = 1e-4
optimizer = optim.Adam(model.parameters(), lr=lr)
loss_fn = nn.CrossEntropyLoss()


for i in range(epoch):
    loss_show = 0
    for j, train_data in enumerate(trainloader):
        optimizer.zero_grad()
        # if Debug_flag == 10:
        #     break
        train_img = train_data[0].to(device)
        label = train_data[1].to(device)

        pred = model(train_img)

        loss = loss_fn(pred, label)
        loss_show += loss
        loss.backward()
        optimizer.step()
        # Debug_flag += 1
    print(loss_show)
