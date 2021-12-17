import argparse
import datetime
import os
import parser

import numpy
import tensorboardX
import torch
import torchvision
from torch.utils import data

import img_augm
#import prepare_dataset
import dataset
from model import Grid
from utils import prepare_sub_folder, write_loss, write_images, denormalize_vgg_adain, put_tensor_cuda

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gf_dim', type=int, default=64)
    parser.add_argument('--df_dim', type=int, default=64)
    parser.add_argument('--dim', type=int, default=3)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--init', type=str, default='kaiming')
    parser.add_argument('--path', type=str, default='/data3/wuxiaolei/models/vgg16-397923af.pth')
    parser.add_argument('--tb_file', type=str, default='/data2/wuxiaolei/project/grid_exp/tensorboard')
    parser.add_argument('--sub_file', type=str, default='new')

    # dataset
    #parser.add_argument('--content_data_path', type=str, default='/data/dataset/Places/data_large')
    parser.add_argument("--content_data_path", type=str, default="/data/dataset/wuxiaolei/COCO2014/train2014")
    parser.add_argument('--style_data_path', type=str, default='/data/dataset/wuxiaolei/WikiArt/train')
    parser.add_argument('--info_path', type=str, default='/data/dataset/wuxiaolei/WikiArt/train_info.csv')

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
    parser.add_argument('--save_freq', type=int, default=50000)

    # loss weight
    parser.add_argument('--clw', type=float, default=1, help='content_weight')
    parser.add_argument('--slw', type=float, default=10, help='style_weight')
    parser.add_argument('--tvw', type=float, default=0.00001, help='tv_loss_weight')

    # bilateral grid
    parser.add_argument('--luma_bins', type=int, default=8)
    parser.add_argument('--channel_multiplier', type=int, default=1)
    parser.add_argument('--spatial_bin', type=int, default=8)
    parser.add_argument('--n_input_channel', type=int, default=256)
    parser.add_argument('--n_input_size', type=int, default=64)
    parser.add_argument('--group_num', type=int, default=16)
    parser.add_argument('--alpha', type=float, default=0.8)
    parser.add_argument('--selection', type=str, default='Ax+b')

    # seed
    parser.add_argument('--seed', type=int, default=123)
    opts = parser.parse_args()

    # fix the seed
    numpy.random.seed(opts.seed)
    torch.manual_seed(opts.seed)
    torch.cuda.manual_seed_all(opts.seed)
    if not os.path.exists(opts.tb_file):
        os.mkdir(opts.tb_file)

    tb_file = opts.tb_file
    sub_file = opts.sub_file
    # Setup logger and output folders
    if not os.path.exists(opts.output_path):
        os.mkdir(opts.output_path)
    output_directory = opts.output_path + sub_file
    checkpoint_directory, image_directory = prepare_sub_folder(output_directory)
    print(checkpoint_directory)
    train_writer = tensorboardX.SummaryWriter(tb_file+sub_file)

    gpu_num = torch.cuda.device_count()

    # prepare dataset
    augmentor = img_augm.Augmentor(crop_size=[opts.image_size, opts.image_size])
    #content_style_dataset = prepare_dataset.PlacesDataset(opts, augmentor)
    content_style_dataset = dataset.ArtDataset(opts, augmentor)
    total_images = len(content_style_dataset)
    print(f"There are total {total_images} image pairs!")
    dataloader = data.DataLoader(content_style_dataset, batch_size=opts.batch_size, num_workers=opts.batch_size, shuffle=True)

    # prepare model
    trainer = Grid(opts, gpu_num).cuda()
    torch.backends.cudnn.benchmark = True

    # start training
    print('-' * 8 + 'Start training' + '-' * 8)
    initial_step = trainer.resume(checkpoint_directory, opts) if opts.resume else 0
    total_step = total_images // opts.batch_size
    step = initial_step
    for iteration in range(opts.epoch):
        for i, data in enumerate(dataloader):
            t0 = datetime.datetime.now()
            step +=1
            input = data
            content_cuda = put_tensor_cuda(input[0])
            style_cuda = put_tensor_cuda(input[1])
            trainer.update_learning_rate()

            # training update
            trainer.update(content_cuda, style_cuda, opts)
            batch_output = trainer.get_output()
            batch_content_style = trainer.get_content_style()
            display = torch.cat([batch_content_style[:1], batch_output[:1]], 3)
            if step % 1000 == 0:
                write_loss(step, trainer, train_writer)
            if step % 1000 == 0:
                write_images('content_style_output', display, train_writer, step)
            if step % 1000 == 0:
                result = torchvision.utils.make_grid(denormalize_vgg_adain(display).cpu())
                torchvision.utils.save_image(result, os.path.join(image_directory, 'test_%08d.jpg' % (total_step + 1)))
            if step % opts.save_freq == 0:
                trainer.save(checkpoint_directory, step)
            t1 = datetime.datetime.now()
            time = t1 - t0
            if step % 50 == 0:
                print("Epoch: %08d/%08d, iteration: %08d/%08d time: %.8s gloss = %.8s  tvloss = %.8f" % (
                    iteration + 1, opts.epoch, step, total_step,
                    time.seconds + 1e-6 * time.microseconds, trainer.gener_loss.item(), trainer.tv_loss.item(),
                    ))
            # if step == 160000:
            #    # trainer.save(checkpoint_directory, step)
            #  print('This iteration takes :{}'.format((time.seconds + 1e-6 * time.microseconds)))
            #  if step == opts.total_steps:
            #      break
        # if step == opts.total_steps:
        #     break
    trainer.save(checkpoint_directory, step)
    print("Training is finished.")
    print("Done.")




