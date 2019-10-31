import os
from PIL import Image
from torchvision import transforms
import torch.utils.data as data


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg"])


def default_loader(path):
    return Image.open(path).convert('RGB')


def single_load(path, config):
    img = default_loader(path)
    img = img.resize(config['img_size'])
    img_tensor = transforms.ToTensor()(img).unsqueeze(0)
    if config['type'] == 64:
        img_tensor = img_tensor.double()
    return img_tensor


class Dataset(data.Dataset):
    def __init__(self, c_dir, s_dir, config):
        super(Dataset, self).__init__()
        self.c_dir = c_dir
        self.s_dir = s_dir
        self.config = config
        self.c_list = [x for x in os.listdir(self.c_dir) if is_image_file(x)]
        self.s_list = [x for x in os.listdir(self.s_dir) if is_image_file(x)]
        min_len = min(len(self.c_list), len(self.s_list))
        self.c_list = self.c_list[0:min_len]
        self.s_list = self.s_list[0:min_len]

    def __getitem__(self, index):
        c_path = os.path.join(self.c_dir, self.c_list[index])
        s_path = os.path.join(self.s_dir, self.s_list[index])

        img = default_loader(c_path)
        img = img.resize(self.config['img_size'])
        c_tensor = transforms.ToTensor()(img)
        if self.config['type'] == 64:
            c_tensor = c_tensor.double()
        c_name = self.c_list[index].split('.')[0]

        img = default_loader(s_path)
        img = img.resize(self.config['img_size'])
        s_tensor = transforms.ToTensor()(img)
        if self.config['type'] == 64:
            s_tensor = s_tensor.double()
        s_name = self.s_list[index].split('.')[0]

        return c_tensor, s_tensor, c_name, s_name

    def __len__(self):
        return len(self.c_list)


class DrawDataset(data.Dataset):
    def __init__(self, img_dir):
        super(DrawDataset, self).__init__()
        self.img_dir = img_dir
        self.img_list = [x for x in os.listdir(self.img_dir)]

    def __getitem__(self, index):
        img_path = os.path.join(self.img_dir, self.img_list[index])

        img = default_loader(img_path)
        img_tensor = transforms.ToTensor()(img)

        return img_tensor

    def __len__(self):
        return len(self.img_list)
