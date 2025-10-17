import cv2
import os
import argparse
import glob
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from models import DnCNN
from utils import *

# 设置CUDA设备
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# 解析命令行参数
parser = argparse.ArgumentParser(description="DnCNN_Test")
parser.add_argument("--num_of_layers", type=int, default=17, help="Number of total layers")
parser.add_argument("--logdir", type=str, default="logs", help='path of log files')
parser.add_argument("--test_data", type=str, default='Set12', help='test on Set12 or Set68')
parser.add_argument("--test_noiseL", type=float, default=25, help='noise level used on test set')
parser.add_argument("--save_dir", type=str, default="results", help='path to save denoised images')  # 新增：保存结果的文件夹
opt = parser.parse_args()

def normalize(data):
    """将图像归一化到0-1范围"""
    return data / 255.

def main():
    # 创建保存结果的文件夹（若不存在）
    save_root = os.path.join(opt.save_dir, opt.test_data)
    os.makedirs(save_root, exist_ok=True)
    print(f"去噪结果将保存至：{save_root}\n")

    # 加载模型
    print('Loading model ...\n')
    net = DnCNN(channels=1, num_of_layers=opt.num_of_layers)
    device_ids = [0]
    model = nn.DataParallel(net, device_ids=device_ids).cuda()
    # 加载训练好的模型参数
    model.load_state_dict(torch.load(os.path.join(opt.logdir, 'net.pth')))
    model.eval()  # 切换到评估模式

    # 加载测试数据
    print('Loading test data ...\n')
    files_source = glob.glob(os.path.join('data', opt.test_data, '*.png'))
    files_source.sort()  # 按文件名排序

    # 处理测试数据并计算PSNR
    psnr_test = 0.0
    for f in files_source:
        # 读取图像（转为灰度图）
        Img = cv2.imread(f)
        Img_gray = np.float32(Img[:, :, 0])  # 取单通道（灰度图）
        Img_norm = normalize(Img_gray)  # 归一化

        # 调整维度为：[batch, channel, height, width]
        Img_expand = np.expand_dims(Img_norm, 0)
        Img_expand = np.expand_dims(Img_expand, 1)
        ISource = torch.Tensor(Img_expand)  # 干净图像

        # 生成噪声并添加到干净图像
        noise = torch.FloatTensor(ISource.size()).normal_(mean=0, std=opt.test_noiseL / 255.)
        INoisy = ISource + noise  # 带噪图像

        # 转移到GPU
        ISource, INoisy = Variable(ISource.cuda()), Variable(INoisy.cuda())

        # 模型推理（去噪）
        with torch.no_grad():  # 关闭梯度计算，节省内存
            Out = torch.clamp(INoisy - model(INoisy), 0., 1.)  # 去噪结果（截断到0-1）

        # 计算PSNR
        psnr = batch_PSNR(Out, ISource, 1.)
        psnr_test += psnr
        print(f"{os.path.basename(f)} PSNR: {psnr:.4f} dB")

        # 保存去噪后的图像
        # 1. 将张量转为numpy数组（CPU上）
        Out_np = Out.cpu().numpy().squeeze()  # 去除多余维度
        # 2. 还原到0-255范围并转为uint8
        Out_np = np.uint8(Out_np * 255.)
        # 3. 构造保存路径
        file_name = os.path.splitext(os.path.basename(f))[0]
        save_path = os.path.join(save_root, f"{file_name}_denoised.png")
        # 4. 保存图像
        cv2.imwrite(save_path, Out_np)

    # 计算平均PSNR
    psnr_test /= len(files_source)
    print(f"\nTest set average PSNR: {psnr_test:.4f} dB")
    print(f"所有去噪图像已保存至：{save_root}")

if __name__ == "__main__":
    main()