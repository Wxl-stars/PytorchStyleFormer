import os
import cv2

import numpy
from PIL import Image
import datetime

import tensorboardX
import data
from utils import normalize_arr_of_imgs, denormalize_vgg, write_2images, write_images
import argparse
from model import Grid
import torch
from collections import namedtuple
from torchvision import transforms
from torch.utils.data import DataLoader
import torchvision
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

from PIL import Image


def save_conv_img(conv_img, sub_filename):
    root = './feature_map'
    if not os.path.exists(root):
        os.mkdir(root)
    sub_file = root + '/' + sub_filename
    if not os.path.exists(sub_file):
        os.mkdir(sub_file)
    conv_img = conv_img.detach().cpu()
    feature_maps = conv_img.squeeze(0)
    img_num = feature_maps.shape[0]  #
    all_feature_maps = []
    for i in range(0, img_num):
        single_feature_map = feature_maps[i, :, :]
        all_feature_maps.append(single_feature_map)
        plt.imshow(single_feature_map)
        plt.savefig(sub_file + '/feature_{}'.format(i))

    sum_feature_map = sum(feature_map for feature_map in all_feature_maps)
    plt.imshow(sum_feature_map)
    plt.savefig(sub_file + "/feature_map_sum.png")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gf_dim', type=int, default=64)
    parser.add_argument('--df_dim', type=int, default=64)
    parser.add_argument('--dim', type=int, default=3)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--init', type=str, default='kaiming')
    parser.add_argument('--path', type=str, default='/data3/wuxiaolei/models/vgg16-397923af.pth')

    # dataset
    parser.add_argument('--root', type=str, default='/data/dataset/')
    parser.add_argument('--input_path', type=str, default='/data2/wuxiaolei/project/content')
    parser.add_argument('--style_path', type=str, default='/data2/wuxiaolei/project/style')
    parser.add_argument('--trained_network', type=str, help="path to the trained network file")
    parser.add_argument('--results_path', type=str, default='./results', help='outputs path')

    # data
    parser.add_argument('--image_size', type=int, default=256)

    # ouptut
    parser.add_argument('--output_path', type=str, default='./output', help='outputs path')

    # train
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--lr_policy', type=str, default='constant', help='step/constant')
    parser.add_argument('--step_size', type=int, default=200000)
    parser.add_argument('--gamma', type=float, default=0.5, help='How much to decay learning rate')
    parser.add_argument('--update_D', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--save_freq', type=int, default=10000)

    # loss weight
    parser.add_argument('--clw', type=float, default=1, help='content_weight')
    parser.add_argument('--slw', type=float, default=10, help='style_weight')
    parser.add_argument('--alpha', type=float, default=1, help='style_weight')

    # bilateral grid
    parser.add_argument('--luma_bins', type=int, default=8)
    parser.add_argument('--channel_multiplier', type=int, default=1)
    parser.add_argument('--spatial_bin', type=int, default=8)
    parser.add_argument('--n_input_channel', type=int, default=256)
    parser.add_argument('--n_input_size', type=int, default=64)
    parser.add_argument('--group_num', type=int, default=16)

    # test
    parser.add_argument('--selection', type=str, default='Ax+b')
    parser.add_argument('--inter_selection', type=str, default='A1x+b2')
    # seed
    parser.add_argument('--seed', type=int, default=123)
    opts = parser.parse_args()

    # fix the seed
    numpy.random.seed(opts.seed)
    torch.manual_seed(opts.seed)
    torch.cuda.manual_seed_all(opts.seed)

    # opts = parser.parse_args()
    options = parser.parse_args()
    if not os.path.exists(options.output_path):
        os.mkdir(options.output_path)
    if not os.path.exists(options.results_path):
        os.mkdir(options.results_path)

    gpu_num = torch.cuda.device_count()

    myNet = Grid(options, gpu_num).cuda()

    initial_step = myNet.resume_eval(options.trained_network)
    torch.backends.cudnn.benchmark = True
    transform_style = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor(),
                                          transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    transform_content = transforms.Compose([transforms.ToTensor(),
                                            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                 std=[0.229, 0.224, 0.225])])
    dataset = data.ImageFolder(options.input_path, transform=transform_content, return_paths=True, resize=False)
    dataset_style = data.ImageFolder(options.style_path, transform=transform_style, return_paths=True, resize=False)
    loader = DataLoader(dataset=dataset, batch_size=1, num_workers=0)
    loader_style = DataLoader(dataset=dataset_style, batch_size=1, num_workers=0)

    step = 0
    total_time = datetime.datetime.now() - datetime.datetime.now()

    for it, images in enumerate(loader):
        for it2, styles in enumerate(loader_style):
            step += 1
            content = images[0]

            style = styles[0]
            content_path = images[1][0]
            style_path = styles[1][0]
            content_name = os.path.split(content_path)[1]
            style_name = os.path.split(style_path)[1]
            style = style.cuda()
            content = content.cuda()
            with torch.no_grad():
                t0 = datetime.datetime.now()
                samp = myNet.sample(content, style)
                t1 = datetime.datetime.now()
            time = t1 - t0
            if step != 1:
                total_time += time
            print("time:%.8s", time.seconds + 1e-6 * time.microseconds)
            print('step:{}'.format(step))
            image_outputs = [denormalize_vgg(samp).clamp_(0., 255.)]

            write_2images(image_outputs, 1, options.output_path, f'{style_name[:-4]}+{content_name[:-4]}+base')

            # write the results
            c = Image.open(content_path)
            style = Image.open(style_path)
            result = Image.open(f'{options.output_path}/{style_name[:-4]}+{content_name[:-4]}+base.png')
            c = c.resize(i for i in result.size)
            s = style.resize((i // 4 for i in result.size))

            new = Image.new(result.mode, (result.width * 2, result.height))
            new.paste(c, box=(0, 0))
            new.paste(result, box=(c.width, 0))
            box = ((c.width // 4) * 3, c.height - s.height)
            new.paste(s, box)
            new.save(f'{options.results_path}/{style_name[:-4]}+{content_name[:-4]}.png', quality=95)
    avag = total_time / (step - 1)
    print("avarge time:%.8s", avag.seconds + 1e-6 * avag.microseconds)
