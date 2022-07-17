import torch
import numpy
from  torch.utils.data import Dataset
class MinistData(Dataset):
    def __init__(self, image_path, label_path):
        _, self.image_num, self.w, self.h, self.images, self.ids = self.get_Minist(image_path, label_path)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        return (self.images[index], self.ids[index])

    def get_Minist(self, image_path, label_path):
        with open(image_path, 'rb') as f:
            s = f.read()
        magic_num = int(s[:4].hex(), 16)
        image_num = int(s[4:8].hex(), 16)
        w = int(s[8:12].hex(), 16)
        h = int(s[12:16].hex(), 16)
        image_dot = [m for m in s[16:]]
        images = numpy.array(image_dot).reshape(-1, 1, 28, 28)
        images = torch.tensor(images, dtype=torch.float)

        with open(label_path, 'rb') as f:
            s = f.read()
        ids = [id for id in s[8:]]
        ids = numpy.asarray(ids)
        ids = torch.tensor(ids, dtype=torch.int64)
        return (magic_num, image_num, w, h, images, ids)