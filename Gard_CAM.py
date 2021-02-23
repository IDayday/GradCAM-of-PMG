import cv2
import os
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
import json
from PIL import Image
from torch.nn import functional as F

    
def preprocess(img_pil):
    """
    input_size: (Tuple:(int,int))    depend on your model's image input
    """
    # normalize = transforms.Normalize(
    # mean=[0.485, 0.456, 0.406],
    # std=[0.229, 0.224, 0.225]
    # )
    # transform = transforms.Compose([
    # transforms.Resize(input_size),
    # transforms.ToTensor(),
    # normalize
    # ])
    # 作者用的以下方法进行transform，我删除了CenterCrop
    transform_test = transforms.Compose([
    transforms.Resize((448,448)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),])
    return transform_test(img_pil)

# 计算grad-cam
def returnGradCAM(feature_map, grads):
    nc, h, w = feature_map.shape
    output_gradcam = []
    gradcam = np.zeros(feature_map.shape[1:], dtype=np.float32)	# feature_map.shape[1:]取第一维度后的尺寸，零初始化
    grads = grads.reshape([grads.shape[0],-1])					# 计算每个通道权重
    weights = np.mean(grads, axis=1)							# 权重均值
    for i, w in enumerate(weights):
        gradcam += w * feature_map[i, :, :]						# 梯度与对应权重相乘再累加
    gradcam = np.maximum(gradcam, 0)                            # 相当于ReLU操作
    gradcam = gradcam / gradcam.max()
    cam_img = np.uint8(255 * gradcam)
    cam_img = cv2.resize(cam_img, (448, 448))
    output_gradcam.append(cam_img)
    return output_gradcam

# 计算concat_grad-cam
def return_concat_GradCAM(feature_map, grads):
    nc, h, w = feature_map.shape
    gradcam = np.zeros(feature_map.shape[1:], dtype=np.float32)	# feature_map.shape[1:]取第一维度后的尺寸，零初始化
    grads = grads.reshape([grads.shape[0],-1])					# 计算每个通道权重
    weights = np.mean(grads, axis=1)							# 权重均值
    for i, w in enumerate(weights):
        gradcam += w * feature_map[i, :, :]						# 梯度与对应权重相乘再累加
    gradcam = cv2.resize(gradcam, (448,448))
    return gradcam

# 由于PMG模型有四组输出，最后一组是整合了前三组的特征进行classifier，所以设计一个一次返回四组卷积特征的函数
def GradCAM(model_name, pretrained, checkpoint_path, device, img_path, label_path, out_name):
    # 存放梯度和特征图
    fmap_block = []
    grad_block = []
    # 定义获取梯度的函数，由于有多个全连接层，所以要依靠反向传播回溯，计算梯度值
    def backward_hook(module, grad_in, grad_out):
        grad_block.append(grad_out[0].detach())

    # 定义获取特征图的函数
    def farward_hook(module, input, output):
        fmap_block.append(output)

    conv_name = ['conv_block1','conv_block2','conv_block3','conv_block_concat']
    net = model_name
    # enumerate可同时获取索引和值
    for index, c in enumerate(conv_name):
        if c != 'conv_block_concat':
            # check the last conv layer name
            # for name, module in net.named_modules():
            #     print('modules:', name)
            # this finalconv_name need to write 
            finalconv_name = c
            # hook 需要锁定的层名称在load前设置好，多gpu训练的模型load后名称会多'module'
            net._modules.get(finalconv_name)[-1].register_forward_hook(farward_hook)
            net._modules.get(finalconv_name)[-1].register_backward_hook(backward_hook) 
            # 默认是多GPU训练
            if pretrained ==True:
                net.eval()

            # 载入待测试的单张图像
            img_pil = Image.open(img_path)
            # 图像三维数据解压为四维，即给一个batch_size=1, (1,3,X,X)
            img_tensor = preprocess(img_pil).unsqueeze(0)
            # PMG有四组预测值
            output = net(img_tensor)
            # label 的json文件
            json_path = label_path
            with open(json_path, 'r') as load_f:
                load_json = json.load(load_f)
            classes = {int(key): value for (key, value) in load_json.items()}
            # print(classes)

            h_x = F.softmax(output[index], dim=1).data.squeeze()
            probs, idx = h_x.sort(0, True)
            probs = probs.numpy()
            idx = idx.numpy()

            # output the prediction
            for i in range(0, 5):
                print('{:.3f} -> {}'.format(probs[i], classes[idx[i]]))

            # backward
            net.zero_grad()
            class_loss = output[index][0,idx]
            class_loss.backward(class_loss.clone().detach())

            # 获取grad
            grads_val = grad_block[index].cpu().data.numpy().squeeze(0) 
            # 当输入图片尺寸太小，或者模型深度太深时，会最终获得(1,C,1,1)的features_map，所以squeeze操作会把后面表示h*w的1也删去，从而报错
            # 由于在for循环中获取farward_hook，虽然每次都clear了内存，但是仍不能清除历史值，所以这里也要给一个索引
            fmap = fmap_block[index].cpu().data.numpy().squeeze(0)
            # 保存cam图片
            gradcam = returnGradCAM(fmap, grads_val)
            print('top1 prediction: %s'%classes[idx[0]])
            img = cv2.imread(img_path[2:])
            height, width, _ = img.shape
            out_file = c + out_name
            heatmap = cv2.applyColorMap(cv2.resize(gradcam[0],(width, height)), cv2.COLORMAP_JET)
            result = heatmap * 0.3 + img * 0.5
            cv2.imwrite(out_file + '.jpg', result)
            # 最后一次不执行清零
            if index < len(conv_name)-2:
                fmap_block.clear()
        else:
            concat = np.zeros((448,448))
            for i in range(len(fmap_block)):
                grads_val = grad_block[i].cpu().data.numpy().squeeze(0)
                fmap = fmap_block[i].cpu().data.numpy().squeeze(0)
                concat_part = return_concat_GradCAM(fmap, grads_val)
                concat += concat_part

            gradcam = np.maximum(concat, 0)                            # 相当于ReLU操作
            gradcam = gradcam / gradcam.max()
            cam_img = np.uint8(255 * gradcam)
            cam_img = cv2.resize(cam_img, (448, 448))

            # 2维ndarray元素数据类型必须转换成np.uint8类型,cv2.applyColorMap()才不会报错
            concat_gradcam = cam_img.astype(np.uint8)
            out_file = c + out_name
            heatmap = cv2.applyColorMap(cv2.resize(concat_gradcam,(width, height)), cv2.COLORMAP_JET)
            result = heatmap * 0.3 + img * 0.5
            cv2.imwrite(out_file + '.jpg', result)



