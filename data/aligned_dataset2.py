import os.path
import random
import torchvision.transforms as transforms
import torch
from data.base_dataset import BaseDataset
from data.image_folder import make_dataset
from PIL import Image
from data.image_folder2 import make_dataset, store_dataset2
from data.base_dataset import BaseDataset, get_transform


class AlignedDataset2(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        self.dir_C = os.path.join(opt.dataroot, opt.phase + 'C')
        self.dir_D = os.path.join(opt.dataroot, opt.phase + 'D')

        self.C_imgs, self.C_paths = store_dataset2(self.dir_C)
        self.D_imgs, self.D_paths = store_dataset2(self.dir_D)

        self.C_size = len(self.C_paths)
        self.D_size = len(self.D_paths)

        transform_list = []

        transform_list += [transforms.ToTensor(),
                           transforms.Normalize((0.5,),
                                                (0.5,))]

        self.transform = transforms.Compose(transform_list)

    def __getitem__(self, index):
        C_path = self.C_paths[index % self.C_size]
        D_path = self.D_paths[index % self.D_size]

        C_img = Image.open(C_path).convert('L')
        D_img = Image.open(D_path).convert('L')

        C_img = self.transform(C_img)
        D_img = self.transform(D_img)

        if self.opt.resize_or_crop == 'no':
            r, g, b = C_img[0] + 1, C_img[1] + 1, C_img[2] + 1
            C_gray = 1. - (0.299 * r + 0.587 * g + 0.114 * b) / 2.
            C_gray = torch.unsqueeze(C_gray, 0)
            input_img = C_img
        else:
            w = C_img.size(2)
            h = C_img.size(1)

            if (not self.opt.no_flip) and random.random() < 0.5:
                idx = [i for i in range(C_img.size(2) - 1, -1, -1)]
                idx = torch.LongTensor(idx)
                C_img = C_img.index_select(2, idx)
                D_img = D_img.index_select(2, idx)
            if (not self.opt.no_flip) and random.random() < 0.5:
                idx = [i for i in range(C_img.size(1) - 1, -1, -1)]
                idx = torch.LongTensor(idx)
                C_img = C_img.index_select(1, idx)
                D_img = D_img.index_select(1, idx)

        return {'C': C_img, 'D': D_img,
                'C_paths': C_path, 'D_paths': D_path}

    def __len__(self):
        return self.C_size

    def name(self):
        return 'AlignedDataset'
