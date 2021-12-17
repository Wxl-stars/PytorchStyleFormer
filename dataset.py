import csv
import random

import cv2
import shutil
import os
import numpy as np
import torch

import warnings

warnings.filterwarnings("error", category=UserWarning)

import torchvision.transforms as transform

from PIL import Image
from utils import normalize_vgg, image_check

Image.MAX_IMAGE_PIXELS = 1e9
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

Image.MAX_IMAGE_PIXELS = 1e9


def image_to_tensor_PIL(path, image_size):
    try:
        image = Image.open(path).convert('RGB')
    except:
        print(f'cropt image:{path}, then remove')
        # shutil.move(path, './')
    # convert a PIL image to tensor (HWC) in range [0,255] to a torch.Tensor(CHW) in the range [0.0,1.0]
    # data_transform = transform.Compose([transform.Resize((image_size, image_size)), transform.ToTensor(),
    #                                     transform.Normalize(mean, std)])
    while(min(image.size[0], image.size[1]))<256:
        image = image.resize((image.size[0]*2, image.size[1]*2))

    alpha=max(image.size[0], image.size[1])/min(image.size[0], image.size[1])
    while(max(image.size[0], image.size[1])>1800 and alpha <3.5):
        image = image.resize((image.size[0]//2, image.size[1]//2))
    # data_transform = transform.Compose([transform.Resize((512, 512)), transform.CenterCrop((image_size, image_size)), transform.ToTensor(),
    #                                     transform.Normalize(mean, std)])
    data_transform = transform.Compose([transform.RandomCrop((image_size, image_size)), transform.ToTensor(),
                                        transform.Normalize(mean, std)])
    image_tensor = data_transform(image)

    return image_tensor


def read_image_PIL(path, image_size):
    try:
        image = Image.open(path).convert('RGB')
    except:
        print(f'cropt image:{path}, then remove')
        #shutil.move(path, './')
    #image = image.resize((image_size, image_size))
    data_transform = transform.Compose([transform.RandomCrop((image_size, image_size))])
    image = data_transform(image)
    return np.array(image)


def read_image_path(file_path, root_path):
    with open(file_path, 'r', encoding='gb18030') as rfile:
        reader_path = csv.DictReader(rfile)
        style_image_paths = []
        for row in reader_path:
            style_image_paths.append(os.path.join(root_path, row['filename']))
    return style_image_paths


IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def make_datast(dir):
    images = []
    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)
    return images


def read_image(image_path, image_size):
    image = cv2.imread(image_path)
    image_tensor = cv2.cvtColor(cv2.resize(image, dsize=(image_size, image_size)), cv2.COLOR_BGR2RGB)
    # image_tensor = torch.from_numpy(image).permute(2, 0, 1)
    return image_tensor


# def image_to_tensor(image):
#     image = np.ascontiguousarray(image, dtype=np.float32)
#     image_tensor = torch.from_numpy(image).permute(2, 0, 1)  # C,H,W
#     # image_tensor = torch.from_numpy(image).permute(2, 0, 1)
#     return image_tensor


def image_to_tensor(image, image_size):
    image_tensor = torch.from_numpy(
        cv2.cvtColor(cv2.resize(image, dsize=(image_size, image_size)), cv2.COLOR_BGR2RGB)).permute(2, 0, 1)  # C,H,W
    # image_tensor = torch.from_numpy(image).permute(2, 0, 1)
    return image_tensor


class ArtDataset():
    def __init__(self, opts, augmentor):
        self.style_paths = read_image_path(opts.info_path, opts.style_data_path)
        self.content_paths = make_datast(opts.content_data_path)
        self.image_size = opts.image_size
        # self.root = opts.style_data_path
        self.aug = augmentor

        self.image_size = opts.image_size

    def __len__(self):
        return len(self.style_paths)

    def __getitem__(self, item):
        style_image = None
        content_index = random.randint(0, len(self.content_paths) - 1)
        content_path = self.content_paths[content_index]

        style_index = random.randint(0, len(self.style_paths) - 1)
        style_path = self.style_paths[style_index]

        # content_image = read_image_PIL(content_path, self.image_size)
        # try:
        #     #style_image = cv2.imread(style_path)
        #     style_image = read_image_PIL(style_path, self.image_size)
        # except:
        #     print(style_path)
        # # content_image = image_check(content_image, content_path, augmentor=self.aug)
        # # style_image = image_check(style_image, style_path, augmentor=self.aug)

        # # content_tensor = normalize_vgg(image_to_tensor(content_image, self.image_size))
        # style_tensdor = normalize_vgg(image_to_tensor(style_image, self.image_size).float())
        content_tensor = image_to_tensor_PIL(content_path, self.image_size)
        style_tensor = image_to_tensor_PIL(style_path, self.image_size)

        return content_tensor, style_tensor
