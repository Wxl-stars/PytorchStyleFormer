import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from mistune import preprocessing

from networks import VGG, decoder, StyleFormer
from utils import get_scheduler, get_model_list, gram_matrix, \
    calc_mean_std, adaptive_instance_normalization, put_tensor_cuda, TVloss, content_loss


class Grid(nn.Module):
    def __init__(self, options, gpu_num):
        super(Grid, self).__init__()
        # build model
        print('-' * 8 + 'init Encoder' + '-' * 8)
        self.vgg = nn.DataParallel(VGG(options), list(range(gpu_num)))
        self.model = nn.DataParallel(StyleFormer(options), list(range(gpu_num)))
        print('-' * 8 + 'init Decoder' + '-' * 8)
        self.decoder = nn.DataParallel(decoder(options), list(range(gpu_num)))

        # Setup the optimizer
        gen_params = list(self.model.parameters()) + list(self.decoder.parameters())
        self.gen_opt = torch.optim.Adam([p for p in gen_params if p.requires_grad], lr=options.lr, betas=(0.8, 0.999),
                                        weight_decay=0.0001, amsgrad=True)

        self.gen_scheduler = get_scheduler(self.gen_opt, options)

        # Loss criteria
        self.mse = nn.MSELoss(reduction='mean')
        self.abs = nn.L1Loss(reduction='mean')
        self.cos = nn.CosineSimilarity()

        self.gram_loss = torch.tensor(0.)
        self.per_loss = torch.tensor(0.)
        self.tv_loss = torch.tensor(0.)

        # image display
        self.input = None
        self.output = None
        self.content_style = None

    def gen_update(self):
        self.gener_loss.backward()
        self.gen_opt.step()

    def update(self, content, style, options):
        # input: content, style

        # zero gradient
        self.gen_opt.zero_grad()

        self.input = input
        self.content_style = torch.cat((content, style), dim=3)

        content_feats = self.vgg(content)
        style_feats = self.vgg(style)

        stylized_feature = self.model(style_feats[-2], content_feats[-2])  # relu3_1

        # stylied photo
        output = self.decoder(stylized_feature)
        self.output = output

        # loss

        # styel loss
        output_feats = self.vgg(output)

        self.gram_loss = self.get_mean_std_diff(output_feats, style_feats)

        # per loss
        self.per_loss = options.clw * content_loss(output_feats[-1], content_feats[-1])
        # self.per_loss = options.clw * self.mse(output_feats[-1], content_feats[-1])

        # # tv loss
        self.tv_loss = TVloss(output, options.tvw)

        # generator total loss
        self.gener_loss = options.slw * self.gram_loss + options.clw * self.per_loss + self.tv_loss
        self.gram_loss = options.slw * self.gram_loss
        self.gen_update()
        return None

    def get_output(self):
        return self.output

    def get_content_style(self):
        return self.content_style

    def get_mean_std_diff(self, feature1, feature2):
        diff = torch.tensor(0.).cuda()
        for i in range(len(feature1)):
            feat1 = feature1[i]
            feat2 = feature2[i]
            feat1_mean, feat1_std = calc_mean_std(feat1)
            feat2_mean, feat2_std = calc_mean_std(feat2)
            diff += self.mse(feat1_mean, feat2_mean) + self.mse(feat1_std, feat2_std)
        return diff

    def update_learning_rate(self):
        if self.gen_scheduler is not None:
            self.gen_scheduler.step()

    def resume(self, checkpoint_dir, options):
        # Load generators
        last_model_name = get_model_list(checkpoint_dir, "gen")
        if last_model_name == None:
            return 0
        state_dict = torch.load(last_model_name)
        self.model.load_state_dict(state_dict['a'])
        self.decoder.load_state_dict(state_dict['b'])
        iterations = int(last_model_name[-11:-3])

        # Load optimizers
        last_model_name = get_model_list(checkpoint_dir, "opt")
        state_dict = torch.load(last_model_name)
        self.gen_opt.load_state_dict(state_dict['a'])

        # Reinitilize schedulers
        self.gen_scheduler = get_scheduler(self.gen_opt, options, iterations)
        print('Resume from iteration %d' % iterations)
        return iterations

    def resume_eval(self, trained_generator):  # 在test的时候都要用什么。。
        state_dict = torch.load(trained_generator)
        self.model.load_state_dict(state_dict['a'])
        self.decoder.load_state_dict(state_dict['b'])
        return 0

    def save(self, snapshot_dir, iterations):
        # Save generators, discriminators, and optimizers
        gen_name = os.path.join(snapshot_dir, 'gen_%08d.pt' % (iterations + 1))
        opt_name = os.path.join(snapshot_dir, 'opt_%08d.pt' % (iterations + 1))
        torch.save({'a': self.model.state_dict(), 'b': self.decoder.state_dict()},
                   gen_name)
        torch.save({'a': self.gen_opt.state_dict()}, opt_name)

    def sample(self, content, style):
        self.eval()
        with torch.no_grad():
            # print('content: ', content.shape)
            # print('style: ', style.shape)
            content_feat = self.vgg(content)
            style_feat = self.vgg(style)

            stylized_feature = self.model(style_feat[-2], content_feat[-2])
            output = self.decoder(stylized_feature)
        self.train()
        return output

    def sample_inter(self, content, style1, style2):
        self.eval()
        with torch.no_grad():
            print('content: ', content.shape)
            print('style1: ', style1.shape)
            print('style2: ', style2.shape)
            content_feat = self.vgg(content)
            style1_feat = self.vgg(style1)
            style2_feat = self.vgg(style2)
            stylized_feature = self.model.module.interpolation(style1_feat[-2], style2_feat[-2], content_feat[-2])
            output = self.decoder(stylized_feature)
        self.train()
        return output

    def sample_mask(self, content, style1, style2, mask):
        self.eval()
        with torch.no_grad():
            print('content: ', content.shape)
            print('style1: ', style1.shape)
            print('style2: ', style2.shape)
            print('mask: ', mask.shape)
            content_feat = self.vgg(content)
            style1_feat = self.vgg(style1)
            style2_feat = self.vgg(style2)
            stylized_feature = self.model(style1_feat[-2], content_feat[-2])
            output1 = self.decoder(stylized_feature)
            stylized_feature = self.model(style2_feat[-2], content_feat[-2])
            output2 = self.decoder(stylized_feature)
            mask = (mask > 0).to(output1.device).float()
            print('mask: ', mask.shape)
            output = mask * output1 + (1 - mask) * output2
        self.train()
        return output

    def test(self, content, style):
        self.eval()
        with torch.no_grad():
            content_feat = self.vgg(content)
            style_feat = self.vgg(style)
            stylized_feature, score_map = self.model(style_feat[-2], content_feat[-2])
            output = self.decoder(stylized_feature)
        self.train()
        return output, score_map

    def feats_crop(self, feats, reference):
        new_feats = []
        for i in range(len(feats)):
            pad = (feats[i].shape[3] - reference[i].shape[3]) // 2
            new_feats.append(feats[i][:, :, pad:-pad, pad:-pad].contiguous())
        return new_feats
