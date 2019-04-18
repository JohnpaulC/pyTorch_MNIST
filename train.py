import torch
import torchvision
from torchvision.datasets import MNIST
from torchvision import transforms
from model import Conv_NN

transform = transforms.Compose([
    # you can add other transformations in this list
    transforms.ToTensor()
])

trainset = MNIST('MNIST', transform = transform)


trainloader = torch.utils.data.DataLoader(trainset, batch_size = 1,
                                        shuffle = True, num_workers = 2)
print(trainset)

print(trainloader)
write_flag = 0

model = Conv_NN()

for i, train_data in enumerate(trainloader):
    if write_flag == 1:
        break
    print(type(train_data))
    train_img = train_data[0].squeeze()
    label = train_data[1]
    print(type(train_img))
    print(train_img.shape)
    print(type(label))

    a = model(train_img)
    print(a)
    write_flag += 1