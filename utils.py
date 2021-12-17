import os

import torch
# torch.set_printoptions(profile="full")
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim import lr_scheduler
import cv2
import numpy as np
from scipy.spatial.distance import cdist



def normalize_arr_of_imgs(arr):
    """
    Normalizes an array so that the result lies in [-1; 1].
    Args:
        arr: numpy array of arbitrary shape and dimensions.
    Returns:
    """
    return arr / 127.5 - 1.


def denormalize_arr_of_imgs(arr):
    """
    Inverse of the normalize_arr_of_imgs function.
    Args:
        arr: numpy array of arbitrary shape and dimensions.
    Returns:
    """
    return (arr + 1.) * 127.5

def normalize_vgg_adain(arr):
    """
    Normalizeds an arry so that the result lies in [0,1]
    """
    return (arr/255.)

def denormalize_vgg_adain(arr):
    return (arr*255.)

def put_tensor_cuda(tensor):
    results = Variable(tensor).cuda()
    return (results)


def prepare_sub_folder(output_directory):
    if not os.path.exists(output_directory):
        os.mkdir(output_directory)
    image_directory = os.path.join(output_directory, 'images')
    if not os.path.exists(image_directory):
        print("Creating directory: {}".format(image_directory))
        os.makedirs(image_directory)
    checkpoint_directory = os.path.join(output_directory, 'checkpoints')
    if not os.path.exists(checkpoint_directory):
        print("Creating directory: {}".format(checkpoint_directory))
        os.makedirs(checkpoint_directory)
    return checkpoint_directory, image_directory


def get_scheduler(optimizer, options, iterations=-1):
    if options.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=options.step_size,
                                        gamma=options.gamma, last_epoch=iterations)
    else:
        scheduler = None  # constant scheduler
    return scheduler


def write_images(str, images_tensor, trainer_writer, step):
    imges_tensor_de = denormalize_vgg(images_tensor)
    imges_tensor_de = imges_tensor_de.clamp(0., 255.)
    image_grid = torchvision.utils.make_grid(imges_tensor_de.cpu(), normalize=True)
    trainer_writer.add_image(str, image_grid, step + 1)


def write_loss(iterations, trainer, train_writer):
    members = [attr for attr in dir(trainer) \
               if not callable(getattr(trainer, attr)) and not attr.startswith("__") and (
                       'loss' in attr or 'grad' in attr or 'nwd' in attr)]
    for m in members:
        train_writer.add_scalar(m, getattr(trainer, m), iterations + 1)


# Get model list for resume
def get_model_list(dirname, key):
    if os.path.exists(dirname) is False:
        return None
    gen_models = [os.path.join(dirname, f) for f in os.listdir(dirname) if
                  os.path.isfile(os.path.join(dirname, f)) and key in f and ".pt" in f]
    if len(gen_models) == 0:
        return None
    gen_models.sort()
    last_model_name = gen_models[-1]
    return last_model_name


def image_check(image, style_path,augmentor=None):
    try:
        h, w = image.shape[0], image.shape[1]
    except:
        print(style_path)
    if max(image.shape) > 500.:
        scale = 500 / max(image.shape)
        image = cv2.resize(image, dsize=(int(scale * w), int(scale * h)))
    if max(image.shape) < 300:
        # Resize the smallest side of the image to 800px
        alpha = 300. / float(min(image.shape))
        if alpha < 4.:
            image = cv2.resize(image, dsize=(int(alpha * h), int(alpha * w)))
        else:
            image = cv2.resize(image, dsize=(300, 300))

    if augmentor is not None:
        image = augmentor(image).astype(np.float32)
    return image


def gram_matrix(tensor):
    # Unwrapping the tensor dimensions into respective variables i.e. batch size, distance, height and width
    _, d, h, w = tensor.size()
    # Reshaping data into a two dimensional of array or two dimensional of tensor
    tensor = tensor.view(d, h * w)
    # Multiplying the original tensor with its own transpose using torch.mm
    # tensor.t() will return the transpose of original tensor
    gram = torch.mm(tensor, tensor.t())
    # Returning gram matrix
    return gram.div(tensor.nelement())


# normalize image to satisfy the vgg
def normalize_vgg(im):
    im /= 255.
    im[0, :, :] -= 0.485
    im[1, :, :] -= 0.456
    im[2, :, :] -= 0.406
    im[0, :, :] /= 0.229
    im[1, :, :] /= 0.224
    im[2, :, :] /= 0.225
    return im


def denormalize_vgg(im):
    im[:, 0, :, :] *= 0.229
    im[:, 1, :, :] *= 0.224
    im[:, 2, :, :] *= 0.225
    im[:, 0, :, :] += 0.485
    im[:, 1, :, :] += 0.456
    im[:, 2, :, :] += 0.406
    im *= 255.
    return im


def calc_mean_std(feat, eps=1e-5):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size()
    assert (len(size) == 4)
    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std_vector = feat_var.sqrt()
    feat_mean_vector = feat.view(N, C, -1).mean(dim=2)
    return feat_mean_vector, feat_std_vector


class LocalGroupNorm(nn.Module):
    def __init__(self, input_channel, G=32, window_size=16):
        super(LocalGroupNorm, self).__init__()
        self.G = G
        self.window_size = window_size

        temp = torch.zeros([input_channel // G, input_channel // G, 1, 1])
        temp_view = temp.view(temp.shape[0], temp.shape[1])
        nn.init.eye_(temp_view)
        temp = temp_view.view_as(temp)
        self.c1_weight = torch.ones([input_channel // G, 1, window_size, window_size]) / (
                window_size * window_size)
        self.c2_weight = temp

    def forward(self, input):
        epsilon = 1e-5
        G = self.G
        N, C, H, W = input.shape[0], input.shape[1], input.shape[2], input.shape[3]
        depth = input.shape[1]
        self.c1_weight = self.c1_weight.cuda(device=input.device)
        self.c2_weight = self.c2_weight.cuda(device=input.device)

        input_reshaped = input.view(N, C // G, G, H, W)

        means = torch.mean(input_reshaped, dim=2)  # N, C//G, H W  1, 8, 64, 64
        means = F.conv2d(F.conv2d(means, weight=self.c1_weight, stride=self.window_size, groups=C // G),
                         weight=self.c2_weight, stride=1)

        means = F.interpolate(means, (input.shape[2], input.shape[3]), mode='bilinear')
        means = means.unsqueeze(2).repeat(1, 1, G, 1, 1)

        stds = (input_reshaped - means).pow(2)
        stds = torch.sqrt(torch.mean(stds, dim=2))
        stds = F.conv2d(F.conv2d(stds, weight=self.c1_weight, stride=self.window_size, groups=C // G),
                        weight=self.c2_weight, stride=1)
        stds = F.interpolate(stds, (input.shape[2], input.shape[3]), mode='bilinear')
        stds = stds.unsqueeze(2).repeat(1, 1, G, 1, 1)

        input = (input_reshaped - means) / torch.sqrt(torch.abs(stds) + epsilon)
        input = input.view(N, C, H, W)
        means = means.view(N, C, H, W)
        stds = stds.view(N, C, H, W)
        return input, means, stds


def adaptive_instance_normalization(content_feat, style_feat):
    assert (content_feat.size()[:2] == style_feat.size()[:2])
    size = content_feat.size()
    style_mean, style_std = calc_mean_std(style_feat)
    content_mean, content_std = calc_mean_std(content_feat)

    style_mean = style_mean.view(size[0], size[1], 1, 1)
    style_std = style_std.view(size[0], size[1], 1, 1)
    content_mean = content_mean.view(size[0], size[1], 1, 1)
    content_std = content_std.view(size[0], size[1], 1, 1)

    normalized_feat = (content_feat - content_mean.expand(
        size)) / content_std.expand(size)
    return normalized_feat * style_std.expand(size) + style_mean.expand(size)


def __write_images(image_outputs, display_image_num, file_name):
    # image_outputs = [images.expand(-1, 3, -1, -1) for images in
    #                  image_outputs]  # expand gray-scale images to 3 channels
    image_tensor = torch.cat([images[:display_image_num] for images in image_outputs], 0)
    image_grid = torchvision.utils.make_grid(image_tensor.data, nrow=display_image_num, padding=0, normalize=True)
    torchvision.utils.save_image(image_grid, file_name, nrow=1)


def write_2images(image_outputs, display_image_num, image_directory, postfix):
    n = len(image_outputs)
    __write_images(image_outputs, display_image_num, '%s/%s.png' % (image_directory, postfix))

def get_hw_2(content_feat, style_feat, g_num):
    device = content_feat.get_device()
    G = g_num
    N, C, H, W = content_feat.shape
    content_vec = content_feat.view(N*G, C//G, -1).permute(0, 2, 1)
    style_vec = style_feat.view(N*G, C//G, -1).permute(0, 2, 1)
    c_numpy = content_vec.detach().cpu().numpy()
    s_numpy = style_vec.detach().cpu().numpy()
    spatial_maps = []
    for i in range(N*G):
        distances = cdist(c_numpy[i], s_numpy[i], metric='cosine')  # 64*64  64*64
        closest = np.argmax(distances, axis=1)
        closest = Variable(torch.from_numpy(closest)).to(device)
        index_i = (closest // 64).view(1, H, W, 1).float()/(H - 1) * 2 - 1
        index_j = (closest % 64).view(1, H, W, 1).float()/(W - 1) * 2 - 1
        spatial_map = torch.cat((index_j, index_i), dim=3)  #  H, W,2
        spatial_maps.append(spatial_map)
    hw_map = torch.cat(spatial_maps, dim=0)  # N, H, W 2
    return hw_map

def get_hw(content_feat, g_num):
     content_feat = content_feat.detach()
     device = content_feat.get_device()

     N, _, H, W = content_feat.shape
     hg, wg = torch.meshgrid([torch.arange(0, H), torch.arange(0, W)]) # [0,511] HxW
     hg = hg.to(device)
     wg = wg.to(device)
     hg = hg.float().repeat(N*g_num, 1, 1).unsqueeze(3) / (H-1) # norm to [0,1] NxHxW
     wg = wg.float().repeat(N*g_num, 1, 1).unsqueeze(3) / (W-1) # norm to [0,1] NxHxW
     hg, wg = hg*2-1, wg*2-1
     hw_map = torch.cat([wg,hg],dim=3)  # NG, H, W,2
     return hw_map


def get_hw3(content_feat, style_feat, g_num):
    content_feat = content_feat.detach()
    style_feat = style_feat.detach()
    device = content_feat.get_device()
    G = g_num
    N, C, H, W = content_feat.shape
    NG = N*G
    content_vec = content_feat.view(NG, C//G, -1)  # NG, C, HW
    style_vec = style_feat.view(NG, C//G, -1)
    multi = content_vec*content_vec
    content_norm = content_vec / (torch.sqrt(torch.sum(content_vec*content_vec, dim=1, keepdim=True)))  # unit vector  NG, C, HW
    style_norm = style_vec / torch.sqrt(torch.sum(style_vec*style_vec, dim=1, keepdim=True))  # NG, C, HW
    closests = []
    for i in range(NG):
        #cosine_dist = 1. - torch.bmm(content_norm[i:i+1, :, :].permute(0,2,1), style_norm[i:i+1, :, :])  # NG, hw, hw
        cosine_dist = torch.bmm(content_norm[i:i+1, :, :].permute(0,2,1), style_norm[i:i+1, :, :])  # NG, hw, hw
        closest = torch.argmax(cosine_dist, dim=2)  # NG, HW
        closests.append(closest)
    closest = torch.cat(closests, dim=0)
    #cosine_dists = 1. - torch.bmm(content_norm.permute(0,2,1), style_norm)  # NG, hw, hw

    # calculate i,j based on index
    index_i = (closest // 64).view(NG, H, W, 1).float() / (H - 1) * 2 - 1
    index_j = (closest % 64).view(NG, H, W, 1).float() / (W - 1) * 2 - 1

    # ######
    # index_i = (closest // 64).view(NG, H, W)
    # index_j = (closest % 64).view(NG, H, W)
    # print(index_i[1])
    # print(index_i[0].shape)
    # print(index_j[1])
    # exit(0)
    hw_map = torch.cat((index_j, index_i), dim=3)  # H, W, 2
    # hw_map_crop = hw_map[:,10:-10, 10:-10, :].permute(0, 3, 1, 2).contiguous()
    # hw_map = F.pad(hw_map_crop, (10,10,10,10), mode='reflect').permute(0,2,3,1).contiguous()
    return hw_map

def TVloss(img, tv_weight):
    """
    Compute total variation loss.
    Inputs:
    - img: PyTorch Variable of shape (1, 3, H, W) holding an input image.
    - tv_weight: Scalar giving the weight w_t to use for the TV loss.
    Returns:
    - loss: PyTorch Variable holding a scalar giving the total variation loss
      for img weighted by tv_weight.
    """
    w_variance = torch.mean(torch.pow(img[:, :, :, :-1] - img[:, :, :, 1:], 2))
    h_variance = torch.mean(torch.pow(img[:, :, :-1, :] - img[:, :, 1:, :], 2))
    loss = tv_weight * (h_variance + w_variance)
    return loss

def content_loss(x, y):
    N,C, _, _ = x.shape
    x_vec = x.view(N,C,-1)
    y_vec = y.view(N,C,-1)
    D_X = pairwise_distances_cos(x_vec, x_vec)
    D_X = D_X/D_X.sum(1, keepdim=True)
    D_Y = pairwise_distances_cos(y_vec, y_vec)
    D_Y = D_Y/D_Y.sum(1, keepdim=True)

    d = torch.abs(D_X-D_Y).mean()
    return d

def pairwise_distances_cos(x, y):
    # x : N,C,-1
    x_norm = x/torch.sqrt((x**2).sum(1,keepdim=True))  #  N, HW
    x_t = x.permute(0, 2, 1)
    # y_t = y.transpose(1,2)
    y_norm = y/torch.sqrt((y**2).sum(1, keepdim=True))

    mul = torch.bmm(x_t, y)

    dist = 1.- mul  #(N, hw*hw)

    return dist
