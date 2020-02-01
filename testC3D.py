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
from config import parse_opts
torch.backends.cudnn.benchmark = True
# opt = parse_opts()
# print(opt)
datass = 'HMDB51'

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
# pathdir = opt.save_path
# os.makedirs(pathdir)

device = torch.device('cuda'if torch.cuda.is_available() else 'cpu')

def conv3x3x3(in_planes, out_planes, stride=1):
    # 3x3x3 convolution with padding
    return nn.Conv3d(in_planes, out_planes, kernel_size=3,
                     stride=stride, padding=1, bias=False)

def downsample_basic_block(x, planes, stride):
    out = F.avg_pool3d(x, kernel_size=1, stride=stride)
    zero_pads = torch.Tensor(out.size(0), planes - out.size(1),
                             out.size(2), out.size(3),
                             out.size(4)).zero_()
    if isinstance(out.data, torch.cuda.FloatTensor):
        zero_pads = zero_pads.cuda()

    out = Variable(torch.cat([out.data, zero_pads], dim=1))

    return out

class ResNeXtBottleneck(nn.Module):
    expansion = 2

    def __init__(self, inplanes, planes, cardinality, stride=1, downsample=None):
        super(ResNeXtBottleneck, self).__init__()
        mid_planes = cardinality * int(planes / 32)
        self.conv1 = nn.Conv3d(inplanes, mid_planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm3d(mid_planes)
        self.conv2 = nn.Conv3d(mid_planes, mid_planes, kernel_size=3, stride=stride,
                               padding=1, groups=cardinality, bias=False)
        self.bn2 = nn.BatchNorm3d(mid_planes)
        self.conv3 = nn.Conv3d(mid_planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class ResNeXt(nn.Module):

    def __init__(self, block, layers, sample_size, sample_duration, shortcut_type='B', cardinality=32, num_classes=400, last_fc=True):
        self.last_fc = last_fc

        self.inplanes = 64
        super(ResNeXt, self).__init__()
        self.conv1 = nn.Conv3d(3, 64, kernel_size=7, stride=(1, 2, 2),
                               padding=(3, 3, 3), bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=2, padding=1)
        self.layer1 = self._make_layer(block, 128, layers[0], shortcut_type, cardinality)
        self.layer2 = self._make_layer(block, 256, layers[1], shortcut_type, cardinality, stride=2)
        self.layer3 = self._make_layer(block, 512, layers[2], shortcut_type, cardinality, stride=2)
        self.layer4 = self._make_layer(block, 1024, layers[3], shortcut_type, cardinality, stride=2)
        last_duration = math.ceil(sample_duration / 16)
        last_size = math.ceil(sample_size / 32)
        self.avgpool = nn.AvgPool3d((last_duration, last_size, last_size), stride=1)
        self.fc = nn.Linear(cardinality * 32 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, shortcut_type, cardinality, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            if shortcut_type == 'A':
                downsample = partial(downsample_basic_block,
                                     planes=planes * block.expansion,
                                     stride=stride)
            else:
                downsample = nn.Sequential(
                    nn.Conv3d(self.inplanes, planes * block.expansion,
                              kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm3d(planes * block.expansion)
                )

        layers = []
        layers.append(block(self.inplanes, planes, cardinality, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, cardinality))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)

        x = x.view(x.size(0), -1)
        if self.last_fc:
            x = self.fc(x)

        return x

def resnet101(**kwargs):
    """Constructs a ResNet-101 model.
    """
    model = ResNeXt(ResNeXtBottleneck, [3, 4, 23, 3], **kwargs)
    return model

def generate_model():
    last_fc = True
    if datass == 'UCF101':
        model = resnet101(num_classes=101, shortcut_type='B', cardinality=32,
                                      sample_size=112, sample_duration=16,
                                      last_fc=last_fc)
    if datass == 'HMDB51':
        model = resnet101(num_classes=51, shortcut_type='B', cardinality=32,
                                      sample_size=112, sample_duration=16,
                                      last_fc=last_fc)
    model = model.cuda()
    model = nn.DataParallel(model, device_ids=None)
    return model

def center_crop(frame):
    frame = frame[8:120, 30:142, :]
    return np.array(frame).astype(np.uint8)

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
            imgm = cv2.resize(img, (112, 112))
            imgxlist.append(imgm)
    #superpixel处理EXTRACT_FOLDER中的文件到slc_folder中
    # if opt.slic == 1:
    #     slic(EXTRACT_FOLDER,slc_folder)
    #     #最后显示时用到slicimg
    #     for i in range(17):
    #         # imgpath = 'E:/JSMA/video-classification-3d-cnn-pytorch-master/tmpspixels'
    #         imgpath = slc_folder
    #         imgname = '/image_%05d.jpg' % (i + 1)
    #         img = cv2.imread(imgpath + imgname)
    #         imgm = cv2.resize(img, (112, 112))
    #         imglist.append(imgm)
    spatial_transform = Compose([Scale(112),
                                 CenterCrop(112),
                                 ToTensor(),
                                 Normalize(get_mean(), [1, 1, 1])])
    temporal_transform = LoopPadding(16)
    #只读入‘’中的文件
    # if opt.slic == 1:
    #     data = Video('tmps', spatial_transform=spatial_transform,
    #                  temporal_transform=temporal_transform,
    #                  sample_duration=16)
    # else:
    data = Video('tmp', spatial_transform=spatial_transform,
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

def main(target,classname,videoname):
    if datass == 'UCF101':
        video_dir = 'E:/UCF101/UCF-101/'
    if datass == 'HMDB51':
        video_dir = 'E:/hmdb51_org/'
    videopath = video_dir+classname+'/'+videoname
    #导入模型
    model = generate_model()
    if datass == 'UCF101':
        model_data = torch.load('E:/JSMA/video-classification-3d-cnn-pytorch-master/resnext-101-kinetics-ucf101_split1.pth')
    if datass == 'HMDB51':
        model_data = torch.load('E:/JSMA/video-classification-3d-cnn-pytorch-master/resnext-101-kinetics-hmdb51_split1.pth')
    optarch = '{}-{}'.format('resnext', 101)
    assert optarch == model_data['arch']
    model.load_state_dict(model_data['state_dict'])
    model.eval()

    #计算输入
    if datass == 'UCF101':
        with open('E:/JSMA/pytorch-video-recognition-master/dataloaders/ucf_labels2.txt', 'r') as f:
            class_names = f.readlines()
            f.close()
    if datass == 'HMDB51':
        with open('E:/JSMA/pytorch-video-recognition-master/dataloaders/hmdb_labels2.txt', 'r') as f:
            class_names = f.readlines()
            f.close()
    if ((classname+'\n') in class_names):
        origin_label = class_names.index(classname+'\n')
    # if opt.slic == 1:
    #     inputs, clip, imglist, imgnoslclist = data(videopath, 'E:/JSMA/video-classification-3d-cnn-pytorch-master/tmp',
    #                                                'E:/JSMA/video-classification-3d-cnn-pytorch-master/tmps')
    # else:
    inputs, clip, imgnoslclist = data(videopath, 'E:/JSMA/video-classification-3d-cnn-pytorch-master/tmp',
                                          'E:/JSMA/video-classification-3d-cnn-pytorch-master/tmps')
    inputs.requires_grad = True
    output = model(inputs)
    probs = torch.nn.Softmax(dim=1)(output)
    label = torch.max(probs, 1)[1].detach().cpu().numpy()[0]
    classname_orig = class_names[label]
    # 保存控制台输出
    # save_dir = pathdir
    # os.makedirs(save_dir,exist_ok=True)
    # txtname = save_dir + '.txt'
    # sys.stdout = Logger(txtname)
    #实施攻击
    print('the origin label is',label,classname_orig)
    #测试攻击时间
    if 50 == label:
        print('success',videoname)

if __name__ == "__main__":
    # txt_path = 'E:/JSMA/video-classification-3d-cnn-pytorch-master/txtUCF101/testlist01.txt'
    txt_path = 'E:/hmdb51_org/test_train_splits/testTrainMulti_7030_splits/wave_test_split2.txt'
    f = open(txt_path,
             'r')
    for lines in f:
        # query_list.append(line.replace('/','').replace('、','').replace(' ','').strip('\n'))
        # 101:ls = lines.strip('\n').replace(' ', '').replace('、', '/').replace('?', '').split('/')
        videoname = lines.split(' ')[0]
        classname = 'wave'
        # classname = ls[0]
        # videoname = ls[1]
        main(None,classname,videoname)