import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
from functools import partial
from torch.autograd.gradcheck import zero_gradients
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
pathdir = 'E:/JSMA/video-classification-3d-cnn-pytorch-master/Testmaxe/epoch100'
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
    model = resnet101(num_classes=101, shortcut_type='B', cardinality=32,
                                      sample_size=112, sample_duration=16,
                                      last_fc=last_fc)

    model = model.cuda()
    model = nn.DataParallel(model, device_ids=None)
    return model


def center_crop(frame):
    frame = frame[8:120, 30:142, :]
    return np.array(frame).astype(np.uint8)

def data(VIDEO_PATH, EXTRACT_FOLDER):

    # read video
    if os.path.exists(VIDEO_PATH):
        video = VIDEO_PATH
    cap = cv2.VideoCapture(video)
    retaining = True

    # clip = []
    imglist = []
    count = 1
    if os.path.exists(EXTRACT_FOLDER):
        shutil.rmtree(EXTRACT_FOLDER)
    if not os.path.exists(EXTRACT_FOLDER):
        os.mkdir(EXTRACT_FOLDER)
    index = 1
    while retaining:
        retaining, img = cap.read()
        if not retaining and img is None:
            continue
        if count != 0 and index <= 17:
            save_path = "{}/image_{:>05d}.jpg".format(EXTRACT_FOLDER, index)
            cv2.imwrite(save_path, img)
            index += 1
        count += 1
        if(count<=17):
            imgm = cv2.resize(img, (112, 112))
            imglist.append(imgm)

    spatial_transform = Compose([Scale(112),
                                 CenterCrop(112),
                                 ToTensor(),
                                 Normalize(get_mean(), [1, 1, 1])])
    temporal_transform = LoopPadding(16)
    data = Video('tmp', spatial_transform=spatial_transform,
                 temporal_transform=temporal_transform,
                 sample_duration=16)
    data_loader = torch.utils.data.DataLoader(data, batch_size=32,
                                              shuffle=False, num_workers=4, pin_memory=True)

    for i, (inputs, segments) in enumerate(data_loader):
        clip = np.transpose(inputs, (0, 2, 3, 4, 1))
        clip = clip[0,:,:,:,:]
        clip = clip.numpy()
        inputs = Variable(inputs,requires_grad = True)
    return inputs,clip,imglist
    # return inputs, img, clip, framelist

def saliency_map(F,x,t,mask,attackflag,idxnum):
    # F 为模型的输出
    # t 为攻击的类别
    # x 表示输入的图像
    # mask 标记位，记录已经访问的点的坐标
    # derivative = torch.autograd.grad(F[0, t], x)
    F[0, t].backward(retain_graph=True)
    derivative = x.grad.data.cpu().numpy().copy()
    # derivative = derivative + 1e-4 * np.random.normal(size=(1,3,16,112,112))
    alphas = derivative * mask  # 预测对攻击目标的贡献
    betas = -np.ones_like(alphas)  # 预测对非攻击目标的贡献
    sal_map = np.abs(alphas) * np.abs(betas) * np.sign(alphas * betas)
    sal_map_frame = sal_map.sum(axis=(0, 1, 3, 4))  # 求每个帧的sal_map
    frameidx = np.argmin(sal_map_frame)# 选择最佳帧
    # print('sal_map_frame', sal_map_frame[frameidx])
    sal_map_frameidx = sal_map[:, :, frameidx, :, :]  # 构建最佳帧的sal_map
    # 寻找像素点
    idxlist = []
    realidxlist = []
    pix_signlist = []
    tempsalmapidx = []
    maskidx = np.ones(shape=sal_map_frameidx.shape)
    if attackflag == 0 or attackflag % 2 == 0:
        for i in range(int(idxnum)):
            idx = np.argmin(sal_map_frameidx)  # 最佳像素和扰动方向
            idx = np.unravel_index(idx, maskidx.shape)  # 转换成最佳帧的(p1,p2)格式
            idxlist.append(idx)
            # print('minsal_map_frameidx', sal_map_frameidx[idx])
            realidx = (idx[0], idx[1], frameidx, idx[2], idx[3])  # 转换成真正的idx
            realidxlist.append(realidx)
            pix_sign = np.sign(alphas)[realidx]
            pix_signlist.append(pix_sign)
            tempsalmapidx.append(sal_map_frameidx[idx])
            sal_map_frameidx[idx] = 1
        for i in range(idxnum):
            sal_map_frameidx[idxlist[i]] = tempsalmapidx[i]
    else:
        for i in range(idxnum):
            idx = np.argmax(sal_map_frameidx)  # 最佳像素和扰动方向
            idx = np.unravel_index(idx, maskidx.shape)  # 转换成最佳帧的(p1,p2)格式
            idxlist.append(idx)
            # print('minsal_map_frameidx', sal_map_frameidx[idx])
            realidx = (idx[0], idx[1], frameidx, idx[2], idx[3])  # 转换成真正的idx
            realidxlist.append(realidx)
            pix_sign = np.sign(alphas)[realidx]
            pix_signlist.append(pix_sign)
            tempsalmapidx.append(sal_map_frameidx[idx])
            sal_map_frameidx[idx] = 1
        for i in range(idxnum):
            sal_map_frameidx[idxlist[i]] = tempsalmapidx[i]

    return frameidx, realidxlist, pix_signlist

#计算PSNR between clean video and adversarial video
def psnr(target, ref):
    import numpy as np
    import math
    # target:目标图像  ref:参考图像  scale:尺寸大小
    # assume RGB image
    target_data = np.array(target)
    # target_data = target_data[scale:-scale, scale:-scale]

    ref_data = np.array(ref)
    # ref_data = ref_data[scale:-scale, scale:-scale]

    diff = ref_data - target_data
    diff = diff.flatten('C')
    rmse = math.sqrt(np.mean(diff ** 2.))
    return 20 * math.log10(1.0 / rmse)

#计算PSNR between clean video and adversarial video
def ssim(X,Y):
    # import cv2
    from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM
    # img1=cv2.imread(imfil1)
    # (h,w)=img1.shape[:2]
    # img2=cv2.imread(imfil2)
    # resized=cv2.resize(img2,(w,h))
    # (h1,w1)=resized.shape[:2]
    # img1=cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    # img2=cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    return ssim( X, Y, data_range=1.0, size_average=False)

def save_img(original_img,diff,save_dir,index):
    adv_img = original_img + diff
    adv_img = adv_img.astype(np.uint8)
    diff = diff.astype(np.uint8)
    if original_img.any() > 1.0:
        original_img = original_img / 255.0
    if adv_img.any() > 1.0:
        adv_img = adv_img / 255.0
    if diff.any() > 1.0:
        diff = diff / 255.0

    saveadv = save_dir + '/' + 'adv' + str(index) + '.png'
    saveimg = save_dir + '/' + 'orig' + str(index) + '.png'

    cv2.imwrite(saveadv, adv_img)
    cv2.imwrite(saveimg, original_img)


#对比展现原始图片和对抗样本图片
def show_images_diff(original_img,original_label,adversarial_img,adversarial_label,diff,save_dir,index):
    plt.figure()
    #归一化
    adv_img = original_img + diff
    adv_img = adv_img.astype(np.uint8)
    diff = diff.astype(np.uint8)
    if original_img.any() > 1.0:
        original_img = original_img/255.0
    if adv_img.any() > 1.0:
        adv_img = adv_img / 255.0
    if diff.any() > 1.0:
        diff = diff / 255.0

    plt.subplot(131)
    plt.title('Original%s'%original_label)
    plt.imshow(original_img)
    plt.axis('off')

    plt.subplot(132)
    plt.title('Adversarial%s'%adversarial_label)
    plt.imshow(adv_img)
    plt.axis('off')

    plt.subplot(133)
    plt.title('difference')
    plt.imshow(diff)
    plt.axis('off')

    plt.tight_layout()
    saveimg = save_dir + '/' + 'frame' + str(index) + '.png'
    plt.savefig(saveimg)
    plt.close()
    # plt.show()

def main(target,classname,videoname):
    video_dir = 'E:/UCF101/'
    videopath = video_dir+classname+'/'+videoname
    #导入模型
    model = generate_model()
    model_data = torch.load('E:/JSMA/video-classification-3d-cnn-pytorch-master/resnext-101-kinetics-ucf101_split1.pth')
    optarch = '{}-{}'.format('resnext', 101)
    assert optarch == model_data['arch']
    model.load_state_dict(model_data['state_dict'])
    model.eval()

    #计算输入
    with open('E:/JSMA/pytorch-video-recognition-master/dataloaders/ucf_labels2.txt', 'r') as f:
        class_names = f.readlines()
        f.close()
    if ((classname+'\n') in class_names):
        origin_label = class_names.index(classname+'\n')
    inputs,clip,imglist = data(videopath,'E:/JSMA/video-classification-3d-cnn-pytorch-master/tmp')
    inputs.requires_grad = True
    output = model(inputs)
    probs = torch.nn.Softmax(dim=1)(output)
    label = torch.max(probs, 1)[1].detach().cpu().numpy()[0]
    classname_orig = class_names[label]
    # 保存控制台输出
    save_dir = pathdir + videoname
    os.makedirs(save_dir,exist_ok=True)
    txtname = save_dir + '/'+ videoname + '.txt'
    sys.stdout = Logger(txtname)
    #实施攻击
    print('the origin label is',label,classname_orig)
    #测试攻击时间
    start =time.clock()
    framelist_end,adv,target,diff,flag = attack_implement(inputs,output,model,target)
    classname_target = class_names[target]
    print('the attack label is',target,classname_target)
    end = time.clock()
    timew = end - start
    print('Running time: %s Seconds' % (timew))
    len_framelist_end = len(framelist_end)

    psnr_list = []
    sm_list = []
    for i in range(len_framelist_end):
        show_images_diff(imglist[framelist_end[i]], origin_label, adv[framelist_end[i], :, :, :], target,diff[framelist_end[i], :, :, :],save_dir,framelist_end[i])
    #保存图片
    for i in range(16):
        save_img(imglist[i],diff[i, :, :, :],save_dir,i)
    list1,list2, count = detection(save_dir)
    #计算psnr和ssim
    for i in range (len(imglist)):
        ps = psnr(adv[i, :, :, :], imglist[i])
        sm = ssim(torch.from_numpy(adv[i, :, :, :]).permute(2, 0, 1).unsqueeze(dim=0),
                  torch.from_numpy(imglist[i]).permute(2, 0, 1).unsqueeze(dim=0))
        psnr_list.append(ps)
        sm_list.append(sm)
    psnr_avg = 0
    ssim_avg = 0
    for i in range(len(sm_list)):
        psnr_avg += psnr_list[i]
        ssim_avg += sm_list[i]
    psnr_avg = psnr_avg / len(sm_list)
    ssim_avg = ssim_avg / len(sm_list)
    return flag,len_framelist_end,timew,diff,psnr_avg,ssim_avg,count

def detection(save_dir):
    #视频对抗样本检测，针对原视频和目标视频进行对抗样本检测
    import foolbox
    import torch
    import torchvision.models as models
    import numpy as np
    import cv2
    import torchvision.transforms as transforms
    import heapq
    import eagerpy as ep
    # instantiate the model
    inception_v3 = models.inception_v3(pretrained=True).eval()
    if torch.cuda.is_available():
        inception_v3 = inception_v3.cuda()
    preprocessing = dict(mean=np.array([0.485, 0.456, 0.406]).reshape((3, 1, 1)).astype(np.float32), std=np.array([0.229, 0.224, 0.225]).reshape((3, 1, 1)).astype(np.float32))
    # mean = torch.from_numpy(np.array([0.485, 0.456, 0.406]).reshape((3, 1, 1)))
    # std = torch.from_numpy(np.array([0.229, 0.224, 0.225]).reshape((3, 1, 1)))
    fmodel = foolbox.models.PyTorchModel(inception_v3, bounds=(0, 1), preprocessing=preprocessing)
    list1 = []
    list2 = []
    # path1 = '/home/vision/video_defend/video_adv-master/video_data/test/BabyCrawling/v_BabyCrawling_g03_c02-0001.jpg'
    path1 = save_dir + '/' + 'orig' + str(0) + '.png'
    img = cv2.imread(path1)
    # img=cv2.imread('E:\\imagenet\\ILSVRC2012_val_00000001.JPEG')
    img = cv2.resize(img, (299, 299))
    # img = img/255.0
    # img = ep.astensors(img)
    transform1 = transforms.Compose([transforms.ToTensor()])
    # img = transform1(img).numpy()
    img = transform1(img)
    vec1 = fmodel(torch.unsqueeze(img.cuda(),dim=0))
    # vec1 = fmodel.predictions(img)
    for index in range(16):
        # t = i
        # s = str(t)
        # s = s.zfill(4)
        path1 = save_dir + '/' + 'adv' + str(index) + '.png'
        # path1 = save_dir + '/' + 'orig' + str(index) + '.png'

        print(path1)
        img0 = cv2.imread(path1)
        # img=cv2.imread('E:\\imagenet\\ILSVRC2012_val_00000001.JPEG')
        img0 = cv2.resize(img0, (299, 299))
        transform1 = transforms.Compose([transforms.ToTensor()])
        # img = transform1(img).numpy()
        img0 = transform1(img0)
        # vec2 = fmodel.predictions(img)
        vec2 = fmodel(torch.unsqueeze(img0.cuda(),dim=0))
        print('predicted class', np.argmax(vec2.cpu))
        list2.append(np.argmax(vec2.cpu))
        dist = np.sqrt(np.sum(np.square(vec1.cpu() - vec2.cpu()).numpy()))
        list1.append(dist)
        vec1 = vec2
    print(list1,list2)
    count = 0
    for i in range(len(list2)):
        if i == 0:
            if list2[i]!=list2[i+1]:
                count += 1
        elif i == len(list2)-1:
            if list2[i]!=list2[i-1]:
                count += 1
        else:
            if list2[i]!=list2[i-1] and list2[i]!=list2[i+1]:
                count += 1
    # return count
    return list1,list2,count


def attack_implement(inputs,output,model,target):
    idxnum = 2
    originputs = inputs.data.cpu().numpy()[0].copy()
    losslist = []
    outputs = model(inputs)
    probs_ = torch.nn.Softmax(dim=1)(outputs)
    label_ = torch.max(probs_, 1)[1].detach().cpu().numpy()[0]
    if target == None:
        for i in range(101):
            target = Variable(torch.Tensor([float(i)]).to(device).long())
            loss_func = torch.nn.CrossEntropyLoss()
            loss = loss_func(outputs, target)
            losslist.append(loss.data.cpu().numpy())
        losslist[label_] = 100
        target = losslist.index(min(losslist))  # 最小值的索引
        print('攻击目标为',target)
    target_label = target  # 攻击目标
    target = Variable(torch.Tensor([float(target_label)]).to(device).long())
    loss_func = torch.nn.CrossEntropyLoss()
    epochs = 800
    theta = 0.1  # 扰动系数
    max_ = originputs[np.unravel_index(np.argmax(originputs), originputs.shape)]  # 定义边界
    min_ = originputs[np.unravel_index(np.argmin(originputs), originputs.shape)]
    print(max_,min_)
    framelist=[]
    framelist_end = []
    num_frame = 0
    flag = 0
    attackflag = 0
    for epoch in range(epochs):
        output = model(inputs)
        probs = torch.nn.Softmax(dim=1)(output)
        label = torch.max(probs, 1)[1].detach().cpu().numpy()[0]
        loss = loss_func(output, target)
        if loss <= 1.3:
            theta = 0.03
        print('epoch={} label={} loss={}'.format(epoch, label, loss))
        if label == target_label:
            flag = 1
            print('framelist',framelist)
            break  # 攻击成功
        zero_gradients(inputs)  # 梯度清零
        mask = np.ones_like(inputs.data.cpu().numpy())
        frame, idxlist, pix_signlist = saliency_map(output, inputs, target_label, mask,attackflag,idxnum)
        print('frame=', frame)
        if frame not in framelist:
            num_frame = num_frame + 1
            framelist_end.append(frame)
        framelist.append(frame)
        j = 0
        # 添加扰动
        for i in range(idxnum):
            # print('idx={}', idxlist[i])
            inputs.data[idxlist[i]] = inputs.data[idxlist[i]] + pix_signlist[i] * theta * (max_ - min_)
            if (inputs.data[idxlist[i]] <= min_) or (inputs.data[idxlist[i]] >= max_):
                j += 1
                print('idx={} over {}'.format(idxlist[i], inputs.data[idxlist[i]]))
                mask[idxlist[i]] = 0
                inputs.data.cpu()[idxlist[i]] = np.clip(inputs.data.cpu()[idxlist[i]], min_, max_)
        if j == idxnum:
            attackflag += 1
            if attackflag >= 10:
                idxnum = int(2 * (attackflag/10 + 1))
    adv = inputs.data.cpu().numpy()[0].copy()
    diff = adv - originputs
    adv = adv.transpose(1, 2, 3, 0)
    diff = diff.transpose(1, 2, 3, 0)
    adv = np.clip(adv, 0, 255).astype(np.uint8)
    print(framelist_end,len(framelist_end))
    return framelist_end,adv,target_label,diff,flag

if __name__ == "__main__":
    timelist = []
    successcount = 0
    countframelist = [0] * 18
    timetotal = 0
    difflist = []
    difftotal = 0
    diffnumlist = []
    diffnumtotal = 0
    psnr_total = 0
    ssim_total = 0
    psnr_avg_list = []
    sm_avg_list = []
    count_list = []
    f = open('E:/JSMA/video-classification-3d-cnn-pytorch-master/txtUCF101/testboundary.txt',
             'r')
    for lines in f:
        # query_list.append(line.replace('/','').replace('、','').replace(' ','').strip('\n'))
        ls = lines.strip('\n').replace(' ', '').replace('、', '/').replace('?', '').split('/')
        classname = ls[0]
        videoname = ls[1]
        flag,len_framelist_end,timew,diff,psnr_avg,sm_avg,count = main(None,classname,videoname)
        #统计结果
        if flag == 1:
            diffx = diff.flatten()
            d = np.linalg.norm(diffx, ord=1, axis=0, keepdims=True)
            perturbation = d / 602112
            difflist.append(perturbation)
            difftotal = difftotal + perturbation
            diffnum = np.flatnonzero(diff).size
            diffnumlist.append(diffnum)
            diffnumtotal = diffnumtotal + diffnum
            successcount += 1
            timelist.append(timew)
            timetotal = timetotal + timew
            psnr_avg_list.append(psnr_avg)
            sm_avg_list.append(sm_avg)
            psnr_total += psnr_avg
            ssim_total += sm_avg
            for i in range(18):
                if len_framelist_end == i:
                    countframelist[i] = countframelist[i] + 1
            count_list.append(count)
        print('successcount=', successcount, 'timelist=', timelist, 'countframelist=', countframelist, 'timetotal=',
              timetotal, 'difflist=', difflist, 'difftotal=', difftotal, 'diffnumlist=', diffnumlist, 'diffnumtotal=',
              diffnumtotal,'psnr_avg_list',psnr_avg_list,'sm_avg_list',sm_avg_list,'psnr_total=',psnr_total,'ssim_total',ssim_total,'countlist',count_list)



