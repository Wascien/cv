from torchvision import transforms
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
import torch

base_path='D:/datasets/CV\PASCAL/VOCdevkit/VOC2012/ImageSets/Segmentation/'
base_image_path='D:/datasets/CV/PASCAL/VOCdevkit/VOC2012/JPEGImages/'
base_label_path='D:/datasets/CV/PASCAL/VOCdevkit/VOC2012/SegmentationClass/'
base_imge_name='{}.jpg'
base_label_name='{}.png'
convert_to_388=transforms.Compose(
    [
        transforms.Resize([388,388]),
    ]
)

convert_to_388_and_pad=transforms.Compose(
    [
            transforms.Resize([388,388]),
            transforms.Pad(padding=92,padding_mode='reflect'),
            transforms.ToTensor()
    ]
)


class SegmentationData(Dataset):

    def __init__(self, txt_path, image_path, label_path):
        super().__init__()
        self.CreateData(txt_path, image_path, label_path)

    def CreateData(self, txt_path, image_path, label_path):
        with open(txt_path, 'r', encoding='utf-8') as f:
            files = f.read().strip().split()

        self.imgs = []
        self.labels = []
        for filename in files:
            img = Image.open(image_path + base_imge_name.format(filename))
            self.imgs.append(convert_to_388_and_pad(img).numpy())

            label = Image.open(label_path + base_label_name.format(filename))
            label = np.array(convert_to_388(label))
            mask = label != 255
            label = mask * label
            self.labels.append(label.flatten())

        imgs = np.array(self.imgs)
        labels = np.array(self.labels)
        self.imgs = torch.Tensor(imgs)
        self.labels = torch.tensor(labels).long()

    def __getitem__(self, index):
        return self.imgs[index], self.labels[index]

    def __len__(self):
        return len(self.imgs)

