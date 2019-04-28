import torch
import cv2 as cv
from model import ConvNN

device = torch.device('cpu' if torch.cuda.is_available() else 'cpu')
model = ConvNN()
model.load_state_dict(torch.load('model.pt', map_location=device))
print(model)

img = cv.imread('number_matrix.bmp')
print(img.shape)

