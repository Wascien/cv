import torch
from torch.utils.data import Dataset
from torchvision import transforms
aug=transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize([227,227]),
    transforms.ToTensor()]
)

class FashionMnist(Dataset):

    def __init__(self, img_path, label_path):
        super().__init__()
        self.load_data_from_path(img_path, label_path)

    def __getitem__(self, index):
        return self.imgs[index], self.labels[index]

    def __len__(self):
        return self.img_num

    def load_data_from_path(self, img_path, label_path):
        with open(img_path, 'rb') as f:
            s = f.read()

        self.img_num = int(s[4:8].hex(), 16)
        self.imgs = torch.FloatTensor(list(iter(s[16:])))
        # print(self.img_num,self.imgs)
        self.imgs = torch.reshape(self.imgs, (-1, 1, 28, 28))
        # print(self.imgs.shape)
        with open(label_path, 'rb') as f:
            s = f.read()
        self.labels = torch.tensor(list(iter(s[8:])))
        # print(self.labels.shape)



def convert_to_227(img,aug):
    return aug(img).numpy().tolist()


def collate_fn(batch):
    inputs, labels = [], []
    for X, y in batch:
        inputs.append(convert_to_227(X, aug))
        labels.append(y)

    return torch.Tensor(inputs), torch.tensor(labels)

