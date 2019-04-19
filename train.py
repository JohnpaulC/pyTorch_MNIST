import torch
import torchvision
from torchvision.datasets import MNIST
from torchvision import transforms
from model import Conv_NN

# At least one transformation
transform = transforms.Compose([
    transforms.ToTensor()
])

trainset = MNIST('MNIST', transform = transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size = 1,
                                        shuffle = True, num_workers = 2)

Debug_flag = 0
model = Conv_NN()

for i, train_data in enumerate(trainloader):
    if Debug_flag == 1:
        break

    train_img = train_data[0].squeeze()
    label = train_data[1]

    a = model(train_img)
    print(a)
    Debug_flag += 1