import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
from functools import partial
from torch.autograd.gradcheck import zero_gradients
import torch
import numpy as np
import cv2
import shutil
import os
import matplotlib.pyplot as plt
from dataset import Video
from mean import get_mean
from spatial_transforms import (Compose, Normalize, Scale, CenterCrop, ToTensor)
from temporal_transforms import LoopPadding
import sys
import time
import parameter
from torchvision import models
from config import parse_opts
torch.backends.cudnn.benchmark = True
# 保存控制台代码到txt


class Logger(object):

    hitCount = 0;
    terminal = None;

    def __init__(self, filename="Default.log"):
        if (Logger.hitCount == 0):
            Logger.terminal = sys.stdout

        Logger.hitCount = Logger.hitCount + 1
        self.log = open(filename, "w")

    def write(self, message):
        Logger.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass
path = os.path.abspath(os.path.dirname(__file__))
type = sys.getfilesystemencoding()
# os.makedirs(pathdir)

device = torch.device('cuda'if torch.cuda.is_available() else 'cpu')

class LSTMModel(nn.Module):

    def __init__(self,original_model,num_classes,hidden_size, fc_size):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size   # 隐藏层数量
        self.num_classes = num_classes   # 类别数
        self.fc_size = fc_size           # 全连接层大小（连接CNN与LSTM）
        # 选择一个特征提取器
        self.features = nn.Sequential(*list(original_model.children())[:-1])   # 特征提取层
        for i, param in enumerate(self.features.parameters()):
            param.requires_grad = False
        self.rnn = nn.LSTM(input_size = fc_size,
                    hidden_size = hidden_size,
                    batch_first = True)                                       # LSTM
        self.fc = nn.Linear(hidden_size, num_classes)                         # logits

        #  CNN+LSTM
    def forward(self, inputs, hidden=None, steps=0):
        # 去掉for循环
        f = self.features(inputs)
        fs = f.squeeze().unsqueeze(0)
        outputs, hidden = self.rnn(fs, hidden)
        outputs = self.fc(outputs)
        outputs = torch.mean(outputs, dim=1)
        # outputs[0, 20].backward(retain_graph=True)
        # dera = inputs.grad.data.cpu().numpy().copy()
        return outputs
# LRCN
def get_model(checkpoint,num_class):
    original_model = models.__dict__['resnet50'](pretrained=False)         # CNN特征提取器
    model = LSTMModel(original_model,num_classes=num_class,hidden_size=512,fc_size=2048)
    model = model.cuda()
    model_info = torch.load(checkpoint)
    model.load_state_dict(model_info['state_dict'])
    return model
datasss = 'HMDB51'
def generate_model_lrcn():
    # assert dataset in ['hmdb51', 'ucf101']
    if datasss == 'HMDB51':
        checkpoint='E:/JSMA/LRCN/checkpoints/hmdb51_save_best.pth'
        num_class=51
    if datasss == 'UCF101':
        checkpoint = 'E:/JSMA/LRCN/checkpoints/ucf101_save_best.pth'
        num_class = 101
    model = get_model(checkpoint,num_class)
    return model

def data(VIDEO_PATH,EXTRACT_FOLDER,slc_folder):
    #读入video
    video = VIDEO_PATH
    imglist = []
    imgxlist = []
    cap = cv2.VideoCapture(video)
    retaining = True
    count = 1
    try:
        shutil.rmtree(EXTRACT_FOLDER)
    except OSError:
        pass
    os.mkdir(EXTRACT_FOLDER)
    index = 1
    #将video处理成图片，只保存17帧在EXTRACT_FOLDER
    while retaining:
        retaining, img = cap.read()
        if not retaining and img is None:
            continue
        if count != 0 and index <= 17:
            save_path = "{}/image_{:>05d}.jpg".format(EXTRACT_FOLDER, index)
            cv2.imwrite(save_path, img)
            index += 1
        count += 1
        #最后显示时用到原img
        if (count <= 17):
            imgm = cv2.resize(img, (224, 224))
            imgxlist.append(imgm)
    #superpixel处理EXTRACT_FOLDER中的文件到slc_folder中
    # if opt.slic == 1:
    #     slic(EXTRACT_FOLDER,slc_folder)
    # #最后显示时用到slicimg
    #     for i in range(17):
    #         # imgpath = 'E:/JSMA/video-classification-3d-cnn-pytorch-master/tmpspixels'
    #         imgpath = slc_folder
    #         imgname = '/image_%05d.jpg' % (i + 1)
    #         img = cv2.imread(imgpath + imgname)
    #         imgm = cv2.resize(img, (224, 224))
    #         imglist.append(imgm)
    spatial_transform = Compose([Scale(224),
                                 CenterCrop(224),
                                 ToTensor(),
                                 Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    temporal_transform = LoopPadding(16)
    #只读入‘’中的文件
    # if opt.slic == 1:
    #     data = Video('tmps', spatial_transform=spatial_transform,
    #              temporal_transform=temporal_transform,
    #              sample_duration=16)
    # else:
    data = Video('tmph', spatial_transform=spatial_transform,
                     temporal_transform=temporal_transform,
                     sample_duration=16)
    data_loader = torch.utils.data.DataLoader(data, batch_size=32,
                                              shuffle=False, num_workers=4, pin_memory=True)

    for i, (inputs, segments) in enumerate(data_loader):
        clip = np.transpose(inputs, (0, 2, 3, 4, 1))
        clip = clip[0, :, :, :, :]
        clip = clip.numpy()
        inputs = Variable(inputs, requires_grad=True)
    # if opt.slic == 1:
    #     return inputs, clip, imglist, imgxlist
    # else:
    return inputs, clip, imgxlist
    # return inputs, img, clip, framelist
def main_white(target,classname,videoname):
    if datasss == 'UCF101':
        video_dir = 'E:/UCF101/UCF-101/'
    if datasss == 'HMDB51':
        video_dir = 'E:/hmdb51_org/'
    videopath = video_dir+classname+'/'+videoname
    #导入模型
    model = generate_model_lrcn()
    #计算输入
    if datasss == 'UCF101':
        with open('E:/JSMA/pytorch-video-recognition-master/dataloaders/ucf_labels2.txt', 'r') as f:
            class_names = f.readlines()
            f.close()
    if datasss == 'HMDB51':
        with open('E:/JSMA/pytorch-video-recognition-master/dataloaders/hmdb_labels2.txt', 'r') as f:
            class_names = f.readlines()
            f.close()
    if ((classname+'\n') in class_names):
        origin_label = class_names.index(classname+'\n')
    # if opt.slic == 1:
    #     inputs, clip, imglist, imgnoslclist= data(videopath,'E:/JSMA/video-classification-3d-cnn-pytorch-master/tmp','E:/JSMA/video-classification-3d-cnn-pytorch-master/tmps')
    # else:
    inputs, clip, imgnoslclist = data(videopath, 'E:/JSMA/video-classification-3d-cnn-pytorch-master/tmph', 'E:/JSMA/video-classification-3d-cnn-pytorch-master/tmphs')
    inputs_tmp = inputs.permute(0, 2, 1, 3, 4)[0, :, :, :, :].cuda()
    outputs = model(inputs_tmp)
    label = torch.max(outputs, 1)[1].detach().cpu().numpy()[0]
    classname_orig = class_names[label]
    # print('the output is',classname_orig)
    if label == origin_label:
        print('success', videoname)

if __name__ == "__main__":
    txt_path = 'C:/Users/大王/Downloads/UCF101TrainTestSplits-RecognitionTask/UCF101-annotation/ucfTrainTestlist/trainlist01.txt'
    f = open(txt_path,
             'r')
    for lines in f:
        # query_list.append(line.replace('/','').replace('、','').replace(' ','').strip('\n'))
        ls = lines.strip('\n').replace(' ', '').replace('、', '/').replace('?', '').split('/')
        classname = ls[0]
        videoname = ls[1]
        # if opt.black == 0:
        main_white(None,classname,videoname)
