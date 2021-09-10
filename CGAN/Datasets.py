from torch.utils.data import Dataset
import numpy as np
import torch
import os
import cv2

class MNIST_dataset(Dataset):
    #初始化数据集
    def __init__(self, root, istrain=True):
        self.dataset = []
        self.sub_dir = "TRAIN" if istrain else "TEST"
        for tag in os.listdir(f"{root}/{self.sub_dir}"):
            img_dir = f"{root}/{self.sub_dir}/{tag}"
            for img_filename in os.listdir(img_dir):
                img_path = f"{img_dir}/{img_filename}"
                self.dataset.append((img_path,tag))

    #获取长度
    def __len__(self):
        return  len(self.dataset)

    #获取数据集内的数据
    def __getitem__(self, item):
        data = self.dataset[item]

        #处理数据X
        img_data = cv2.imread(data[0], 0)
        img_data = img_data.reshape(-1, 28, 28)
        # img_data = np.swapaxes(img_data,0,1)


        #把数据变成一维的  hwc转换成v
        # img_data = img_data.reshape(-1)
        img_data = np.float32(img_data)

        #归一化
        img_data = img_data / 255

        #onehot
        # tag_one_hot = np.zeros(1, 10)
        # tag_one_hot = torch.zeros(1, 10)
        # tag_one_hot[..., int(float(data[1]))] = 1
        tag_one_hot = torch.zeros(10)
        tag_one_hot[int(float(data[1]))] = 1

        # return torch.Tensor(np.float32(img_data)), torch.Tensor(np.float32(tag_one_hot))
        return torch.Tensor(np.float32(img_data)), tag_one_hot

