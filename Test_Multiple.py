import torch
import cv2 as cv
from model import ConvNN


device = torch.device('cpu' if torch.cuda.is_available() else 'cpu')
model = ConvNN()
model.load_state_dict(torch.load('model.pt', map_location=device))
print(model)

img = cv.imread('number_matrix.bmp')
img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

img = torch.from_numpy(img).unsqueeze(0).float() / 255.0

img_size = 28
_, h, w = img.shape
index_h = int(h / img_size)
index_w = int(w / img_size)

for i in range(index_w):
    for j in range(index_h):
        image = img[:, i * img_size : (i + 1) * img_size, j * img_size : (j + 1) * img_size]
        print(image.shape)
        image = image.unsqueeze(0)
        print(image.shape)
        _, pred = model(image, apply_softmax=True).max(dim=1)
        print(pred)

        image_show = image.squeeze().permute(1, 2, 0).numpy()
        cv.imshow('fuck', image_show)
        cv.waitKey()
        cv.destroyWindow('fuck')