import torch
import cv2
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import numpy as np


VggNet = nn.Sequential(
    nn.Conv2d(3, 3, (1, 1)),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(3, 64, (3, 3)),
    nn.ReLU(),  # relu1-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 64, (3, 3)),
    nn.ReLU(),  # relu1-2
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 128, (3, 3)),
    nn.ReLU(),  # relu2-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 128, (3, 3)),
    nn.ReLU(),  # relu2-2
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 256, (3, 3)),
    nn.ReLU(),  # relu3-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),  # relu3-2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),  # relu3-3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),  # relu3-4
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 512, (3, 3)),
    nn.ReLU(),  # relu4-1, this is the last layer used
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu4-2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu4-3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu4-4
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu5-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu5-2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu5-3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU()  # relu5-4
)


vgg16 = nn.Sequential(
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(3, 64, kernel_size=3, stride=1),
    nn.ReLU(inplace=True),  # relu1_1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 64, kernel_size=3, stride=1),
    nn.ReLU(inplace=True),
    nn.MaxPool2d(kernel_size=2, stride=2,padding=0, dilation=1, ceil_mode=False),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 128, kernel_size=3, stride=1),
    nn.ReLU(inplace=True),  # relu2_1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128,128, kernel_size=3, stride=1),
    nn.ReLU(inplace=True),
    nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 256, kernel_size=3,stride=1),
    nn.ReLU(inplace=True),  # relu3_1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, kernel_size=3, stride=1),
    nn.ReLU(inplace=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, kernel_size=3, stride=1),
    nn.ReLU(inplace=True),
    nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 512, kernel_size=3, stride=1),
    nn.ReLU(inplace=True),  # relu4_1
)

vgg_conv_list = [1,4,8,11,15,18,21,25]
vgg_model_conv_list = [0, 2, 5, 7, 10, 12, 14, 17]

class VGG(nn.Module):
    def __init__(self, options):
        super(VGG, self).__init__()
        # vgg_pad
        vgg_model = models.vgg16(pretrained=False)
        vgg_model.load_state_dict(torch.load(options.path))
        vgg_model = vgg_model.features
        vgg = vgg16

        for i in range(8):
            vgg[vgg_conv_list[i]].weight = vgg_model[vgg_model_conv_list[i]].weight
            vgg[vgg_conv_list[i]].bias = vgg_model[vgg_model_conv_list[i]].bias
        self.test = vgg[vgg_conv_list[7]].weight

        for p in self.parameters():
            p.requires_grad = False
        self.slice1 = vgg[:3]  # relu1_1
        self.slice2 = vgg[3:10]  # relu2_1
        self.slice3 = vgg[10:17]  # relu3_1
        self.slice4 = vgg[17:27]  # relu4_1
        for p in self.parameters():
            p.requires_grad = False

    def forward(self, x):
        out = []
        x = self.slice1(x)
        out.append(x)
        x = self.slice2(x)
        out.append(x)
        x = self.slice3(x)
        out.append(x)
        x = self.slice4(x)
        out.append(x)
        return out


class decoder(nn.Module):
    def __init__(self, options):
        super(decoder, self).__init__()
        self.model = nn.Sequential(
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(256, 256, (3, 3)),
            nn.ReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(256, 256, (3, 3)),
            nn.ReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(256, 256, (3, 3)),
            nn.ReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(256, 128, (3, 3)),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(128, 128, (3, 3)),
            nn.ReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(128, 64, (3, 3)),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(64, 64, (3, 3)),
            nn.ReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(64, 3, (3, 3)),
        )

    def forward(self, stylized_feature):
        x = self.model(stylized_feature)
        return x


class ConvBlock(nn.Module):
    def __init__(self, inc, outc, kernel_size=3, padding=1, stride=1, use_bias=True, activation=nn.ReLU,
                 batch_norm=False):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(int(inc), int(outc), kernel_size, stride=stride, bias=use_bias)
        # self.activation = activation() if activation else None
        if activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif activation is None:
            self.activation = None
        else:
            self.activation = activation()
        self.bn = nn.BatchNorm2d(outc) if batch_norm else None
        self.p = padding

    def forward(self, x):
        x = F.pad(x, (self.p, self.p, self.p, self.p), mode='reflect')
        x = self.conv(x)
        if self.bn:
            x = self.bn(x)
        if self.activation:
            x = self.activation(x)
        return x


class Coeffs(nn.Module):

    def __init__(self, nin=16, nout=17, options=None):
        super(Coeffs, self).__init__()
        self.nin = nin
        self.nout = nout

        self.lb = options.luma_bins  #8#params['luma_bins']
        self.cm = options.channel_multiplier  #1#params['channel_multiplier']
        self.sb = options.spatial_bin  #8#params['spatial_bin']
        self.G = options.group_num  #16
        bn = False
        nsize = options.n_input_size  #64#params['net_input_size']
        nchannel = options.n_input_channel  #256
        self.relu = nn.ReLU()

        n_layers_splat = int(np.log2(nsize / self.sb))
        self.splat_features = nn.ModuleList()
        prev_ch = nchannel  #nin
        for i in range(n_layers_splat):
            use_bn = False  #bn if i > 0 else False
            self.splat_features.append(ConvBlock(prev_ch, nchannel, 3, stride=2, batch_norm=use_bn))
            prev_ch = nchannel

        # local features
        self.local_features = nn.ModuleList()
        self.local_features.append(ConvBlock(nchannel, 32 * self.cm * self.lb, 3, stride=1, batch_norm=bn))
        self.local_features.append(ConvBlock(32 * self.cm * self.lb, 32 * self.cm * self.lb, 3, stride=1, activation=None, use_bias=False))

        # predicton
        self.conv_out = ConvBlock(32 * self.cm * self.lb, self.G * self.lb * nout * nin, 1, padding=0, activation=None)

    def forward(self, lowres_input):
        bs = lowres_input.shape[0]

        x = lowres_input
        for layer in self.splat_features:
            x = layer(x)
        splat_features = x

        x = splat_features
        for layer in self.local_features:
            x = layer(x)
        local_features = x

        fusion_grid = local_features
        fusion = self.relu(fusion_grid)

        x = self.conv_out(fusion)
        s = x.shape
        x = x.view(bs * self.G, self.nin * self.nout, self.lb, s[2], s[3])  # B x Coefs x Luma x Spatial x Spatial
        return x


class GuideNN(nn.Module):
    def __init__(self, options=None):
        super(GuideNN, self).__init__()
        self.conv1 = ConvBlock(16, 4, kernel_size=1, padding=0)
        self.conv2 = ConvBlock(4, 1, kernel_size=1, padding=0, activation='sigmoid')
        self.G = options.group_num

    def forward(self, x):
        x = self.conv2(self.conv1(x))
        return x


class Slice(nn.Module):
    def __init__(self):
        super(Slice, self).__init__()

    def forward(self, affine_transformation, guidemap):
        device = affine_transformation.get_device()

        N, _, H, W = guidemap.shape
        hg, wg = torch.meshgrid([torch.arange(0, H), torch.arange(0, W)]) # [0,511] HxW
        if device >= 0:
            hg = hg.to(device)
            wg = wg.to(device)
        hg = hg.float().repeat(N, 1, 1).unsqueeze(3) / (H-1) # norm to [0,1] NxHxWx1
        wg = wg.float().repeat(N, 1, 1).unsqueeze(3) / (W-1) # norm to [0,1] NxHxWx1
        hg, wg = hg*2-1, wg*2-1
        guidemap = guidemap.permute(0, 2, 3, 1).contiguous()
        guidemap_guide = torch.cat([wg, hg, guidemap], dim=3).unsqueeze(1) # Nx1xHxWx3
        coeff = F.grid_sample(affine_transformation, guidemap_guide, mode='bilinear', padding_mode='reflection', align_corners=True)
        return coeff.squeeze(2)


class ApplyCoeffs(nn.Module):
    def __init__(self, options=None):
        super(ApplyCoeffs, self).__init__()
        self.degree = 3
        self.G = options.group_num
        self.alpha = options.alpha
        self.sect = options.selection
        self.inter_sect = options.inter_selection
        # self.sect = 'Ax+b'

    def forward(self, coeff, full_res_input):
        N, C, H, W = full_res_input.shape
        CG = C // self.G
        output = []
        for i in range(self.G):
            # print(full_res_input.shape)
            # print(coeff.shape)
            # print(self.sect)
            if self.sect == 'Ax+b':
                x = torch.sum(full_res_input * coeff[:, i*(CG+1):(i+1)*(CG+1)-1, :, :], dim=1, keepdim=True) + coeff[:, (i+1)*(CG+1)-1:(i+1)*(CG+1), :, :]
            if self.sect == 'Ax':
                x = torch.sum(full_res_input * coeff[:, i*(CG+1):(i+1)*(CG+1)-1, :, :], dim=1, keepdim=True)  # no bias
            if self.sect == 'x+b':
                x = torch.sum(full_res_input, dim=1, keepdim=True) + coeff[:, (i+1)*(CG+1)-1:(i+1)*(CG+1), :, :]
            if self.sect == 'b':
                x = coeff[:, (i+1)*(CG+1)-1:(i+1)*(CG+1), :, :]
            if self.sect == 'aAx+b':
                x = torch.sum(full_res_input * self.alpha*coeff[:, i*(CG+1):(i+1)*(CG+1)-1, :, :], dim=1, keepdim=True) + coeff[:, (i+1)*(CG+1)-1:(i+1)*(CG+1), :, :]
            output.append(x)
        return torch.cat(output, dim=1)

    def mix(self, coeff1, coeff2, full_res_input):
        N, C, H, W = full_res_input.shape
        CG = C // self.G
        output = []
        a = 0.9
        b = 0.1
        for i in range(self.G):
            # print(self.inter_sect)
            if self.inter_sect == 'A1x+b2':
                x = torch.sum(full_res_input * coeff1[:, i*(CG+1):(i+1)*(CG+1)-1, :, :], dim=1, keepdim=True) + coeff2[:, (i+1)*(CG+1)-1:(i+1)*(CG+1), :, :]  # A1xx+b2
            if self.inter_sect == 'A2x+b2':
                x = torch.sum(full_res_input * coeff2[:, i*(CG+1):(i+1)*(CG+1)-1, :, :], dim=1, keepdim=True) + coeff2[:, (i+1)*(CG+1)-1:(i+1)*(CG+1), :, :]  # A2x+b2
            if self.inter_sect == 'A2x+b1':
                x = torch.sum(full_res_input * coeff2[:, i*(CG+1):(i+1)*(CG+1)-1, :, :], dim=1, keepdim=True) + coeff1[:, (i+1)*(CG+1)-1:(i+1)*(CG+1), :]
            if self.inter_sect == '(a1A1+a2A2)x+b1':
                x = torch.sum(full_res_input * (a*coeff1[:, i*(CG+1):(i+1)*(CG+1)-1, :, :]+b*coeff2[:, i*(CG+1):(i+1)*(CG+1)-1, :, :]), dim=1, keepdim=True) + coeff1[:, (i+1)*(CG+1)-1:(i+1)*(CG+1), :]
            if self.inter_sect == '(a1A1+a2A2)x+b2':
                x = torch.sum(full_res_input * (a*coeff1[:, i*(CG+1):(i+1)*(CG+1)-1, :, :]+b*coeff2[:, i*(CG+1):(i+1)*(CG+1)-1, :, :]), dim=1, keepdim=True) + coeff2[:, (i+1)*(CG+1)-1:(i+1)*(CG+1), :]
            if self.inter_sect == '(a1A1+a2A2)x+a1*b1+a2*b2 ':
                x = torch.sum(full_res_input * (a*coeff1[:, i*(CG+1):(i+1)*(CG+1)-1, :, :]+b*coeff2[:, i*(CG+1):(i+1)*(CG+1)-1, :, :]), dim=1, keepdim=True) + a*coeff1[:, (i+1)*(CG+1)-1:(i+1)*(CG+1), :] + b*coeff2[:, (i+1)*(CG+1)-1:(i+1)*(CG+1),:]
            output.append(x)
        return torch.cat(output, dim=1)


class StyleFormer(nn.Module):
    def __init__(self, options):
        super(StyleFormer, self).__init__()
        self.coeffs = Coeffs(options=options)
        self.att = AttModule(options)
        self.guide = GuideNN(options=options)
        self.slice = Slice()
        self.apply_coeffs = ApplyCoeffs(options=options)
        self.G = options.group_num
        self.fcc = nn.Conv2d(256, 256, 1, 1)
        self.fcs = nn.Conv2d(256, 256, 1, 1)

    def forward(self, style_feat, content_feat):
        coeffs = self.coeffs(style_feat)
        content_feat = self.fcc(content_feat)
        style_feat = self.fcs(style_feat)
        content_feat = F.group_norm(content_feat, num_groups=self.G)
        style_norm = F.group_norm(style_feat, num_groups=self.G)
        N, C, H, W = content_feat.shape
        content_feat = content_feat.view(N*self.G, C//self.G, H, W)
        N, C, Hs, Ws = style_feat.shape
        style_feat = style_feat.view(N*self.G, -1, Hs, Ws)
        style_norm = style_norm.view(N*self.G, -1, Hs, Ws)

        # grid with attention
        att_coeffs = self.att(content_feat, style_norm, coeffs)
        guide = self.guide(content_feat)

        slice_coeffs = self.slice(att_coeffs, guide)
        out = self.apply_coeffs(slice_coeffs, content_feat)
        out = out.view(N, C, H, W)
        return out

    def interpolation(self, style1_feat, style2_feat, content_feat):
        coeff1 = self.coeffs(style1_feat)
        coeff2 = self.coeffs(style2_feat)

        content_feat = self.fcc(content_feat)
        style1_feat = self.fcs(style1_feat)
        style2_feat = self.fcs(style2_feat)

        content_feat = F.group_norm(content_feat, num_groups=self.G)
        style1_norm = F.group_norm(style1_feat, num_groups=self.G)
        style2_norm = F.group_norm(style2_feat, num_groups=self.G)
        N, C, H, W = content_feat.shape
        content_feat = content_feat.view(N*self.G, C//self.G, H, W)
        N, C, Hs1, Ws1 = style1_feat.shape
        N, C, Hs2, Ws2 = style2_feat.shape
        style1_feat = style1_feat.view(N*self.G, -1, Hs1, Ws1)
        style1_norm = style1_feat.view(N*self.G, -1, Hs1, Ws1)
        style2_feat = style2_norm.view(N*self.G, -1, Hs2, Ws2)
        style2_norm = style2_norm.view(N*self.G, -1, Hs2, Ws2)

        # grid with attention
        att_coeffs1 = self.att(content_feat, style1_norm, coeff1)
        att_coeffs2 = self.att(content_feat, style2_norm, coeff2)
        guide = self.guide(content_feat)

        # interpolation
        slice_coeffs1 = self.slice(att_coeffs1, guide)
        slice_coeffs2 = self.slice(att_coeffs2, guide)

        # style mix
        out = self.apply_coeffs.mix(slice_coeffs1, slice_coeffs2, content_feat)
        out = out.view(N, C, H, W)
        return out

    def mask_mix(self, style1_feat, style2_feat, content_feat):
        coeff1 = self.coeffs(style1_feat)
        coeff2 = self.coeffs(style2_feat)

        content_feat = self.fcc(content_feat)
        style1_feat = self.fcs(style1_feat)
        style2_feat = self.fcs(style2_feat)

        content_feat = F.group_norm(content_feat, num_groups=self.G)
        style1_norm = F.group_norm(style1_feat, num_groups=self.G)
        style2_norm = F.group_norm(style2_feat, num_groups=self.G)
        N, C, H, W = content_feat.shape
        content_feat = content_feat.view(N*self.G, C//self.G, H, W)
        N, C, Hs1, Ws1 = style1_feat.shape
        N, C, Hs2, Ws2 = style2_feat.shape
        style1_feat = style1_feat.view(N*self.G, -1, Hs1, Ws1)
        style1_norm = style1_feat.view(N*self.G, -1, Hs1, Ws1)
        style2_feat = style2_norm.view(N*self.G, -1, Hs2, Ws2)
        style2_norm = style2_norm.view(N*self.G, -1, Hs2, Ws2)

        # grid with attention
        att_coeffs1 = self.att(content_feat, style1_norm, coeff1)
        att_coeffs2 = self.att(content_feat, style2_norm, coeff2)
        guide = self.guide(content_feat)

        # interpolation
        slice_coeffs1 = self.slice(att_coeffs1, guide)
        slice_coeffs2 = self.slice(att_coeffs2, guide)

        # style mix
        out = self.apply_coeffs.mix(slice_coeffs1, slice_coeffs2, content_feat)
        out = out.view(N, C, H, W)
        return out


class AttModule(nn.Module):
    def __init__(self, options):
        super(AttModule, self).__init__()
        self.convc1 = ConvBlock(16, 16, stride=2)
        self.convc2 = ConvBlock(16, 16, stride=2, activation=None)

        self.convs1 = ConvBlock(16, 16, stride=2)
        self.convs2 = ConvBlock(16, 16, stride=2, activation=None)

        self.grid_channel = options.luma_bins*16*17
        self.convsr = ConvBlock(self.grid_channel, self.grid_channel, activation=None)
        self.G = options.n_input_channel//options.group_num
        self.cpg = options.group_num  # channels per group

        self.sp = options.spatial_bin

    def forward(self, c, s, grid):
        Ng, Cg, Lg, Hg, Wg = grid.shape
        Ng, C, Hs, Ws = s.shape

        c = self.convc1(c)
        c1 = self.convc2(c)
        Ng, C, H, W = c1.shape
        c1 = c1.view(Ng, 16, -1)

        s = self.convs1(s)
        s1 = self.convs2(s).view(Ng, 16, -1)

        cs = torch.bmm(c1.permute(0, 2, 1), s1)  # attention map

        # ############# visualization
        # for i in range(16):
        #     display = cs.view(16, H, W, 16, 16)[i, :, :, 14, 13]
        #     mx = torch.max(display)
        #     mn = torch.min(display)
        #     display = (display - mn) / (mx - mn)
        #     # display = F.interpolate(display, (64, 64))
        #     heatmap = display.cpu().detach().numpy()
        #     heatmap = heatmap*255
        #     heatmap=heatmap.astype(np.uint8)
        #     heatmap = cv2.resize(heatmap, (W*16, H*16))
        #     #cv2.imwrite('map.png', heatmap)
        #     heatmap=cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        #     cv2.imwrite(f'attention_results/heatmap{i+1}.png', heatmap)
        # exit(0)
        ########### visualization end

        cs = F.softmax(cs, dim=2)
        grid = grid.view(Ng, -1, Hg, Wg)
        sr = self.convsr(grid).view(Ng, self.grid_channel, -1)
        rs = torch.bmm(sr, cs.permute(0,2,1))
        return rs.view(Ng, Cg, Lg, H, W)



