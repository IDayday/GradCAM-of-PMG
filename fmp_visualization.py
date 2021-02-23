from tools.Gard_CAM import *
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from model import *
from Resnet import *
import numpy as np
import os
from utils import *

model_path='./bird/model.pth'
net = torch.load(model_path)
model_name = net
pretrained = True
checkpoint_path = model_path
device = 'cpu'
label_path = './CUB_200_2011/CUB_200_2011/CUB_200_2011/labels.json'
img_path = './9.jpg'
out_name = '_class003_9'
"""
# 预测单张图片
# img_pil = Image.open(img_path)
# img_tensor = transform_test(img_pil).unsqueeze(0)
# pred_1, pred_2, pred_3, pred_concat = net(img_tensor)
"""
"""
# 获取具体的预测值
_, pred_1 = torch.max(pred_1.data, 1)
_, pred_2 = torch.max(pred_2.data, 1)
_, pred_3 = torch.max(pred_3.data, 1)
_, pred_concat = torch.max(pred_concat.data, 1)
print(pred_1,pred_2,pred_3,pred_concat)
"""

cam = GradCAM(model_name, pretrained, checkpoint_path, device, img_path, label_path, out_name)
