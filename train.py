import torch
import torchvision
from torchvision.datasets import MNIST
from torchvision import transforms

transform = transforms.Compose([
    # you can add other transformations in this list
    transforms.ToTensor()
])

trainset = MNIST('MNIST', transform = transform)


trainloader = torch.utils.data.DataLoader(trainset, batch_size = 10,
                                        shuffle = True, num_workers = 2)
print(trainset)

print(trainloader)
write_flag = 0

for i, train_data in enumerate(trainloader):
    if write_flag == 1:
        break
    print(type(train_data))
    train_img = train_data[0]
    label = train_data[1]
    print(type(train_img))
    print(train_img.shape)
    print(type(label))
    write_flag += 1