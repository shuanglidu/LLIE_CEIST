import numpy as np
import torch
from torch import nn
import os
from collections import OrderedDict
from torch.autograd import Variable
import util.util as util
from collections import OrderedDict
from torch.autograd import Variable
import itertools
import util.util as util
from util.image_pool import ImagePool
from .base_model import BaseModel
import random
from . import networks
import sys


class SingleModel(BaseModel):
    def name(self):
        return 'SingleGANModel'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)

        nb = opt.batchSize
        size = opt.fineSize
        self.opt = opt
        self.input_rgblow = self.Tensor(nb, opt.input_nc, size, size)
        self.input_rgbnorm = self.Tensor(nb, opt.input_nc, size, size)
        self.input_attlow = self.Tensor(nb, 1, size, size)
        self.input_attnorm = self.Tensor(nb, 1, size, size)

        if opt.vgg > 0:
            self.vgg_loss = networks.PerceptualLoss(opt)
            if self.opt.IN_vgg:
                self.vgg_patch_loss = networks.PerceptualLoss(opt)
                self.vgg_patch_loss.cuda()
            self.vgg_loss.cuda()
            self.vgg = networks.load_vgg16("./model", self.gpu_ids)
            self.vgg.eval()
            for param in self.vgg.parameters():
                param.requires_grad = False
        elif opt.fcn > 0:
            self.fcn_loss = networks.SemanticLoss(opt)
            self.fcn_loss.cuda()
            self.fcn = networks.load_fcn("./model")
            self.fcn.eval()
            for param in self.fcn.parameters():
                param.requires_grad = False
        # load/define networks

        self.netAtt = networks.define_attUnet(opt.input_attnc, opt.output_attnc, opt.which_model_net1, self.gpu_ids,)

        skip = True if opt.skip > 0 else False
        self.netG_A = networks.define_G(opt.input_nc, opt.output_nc,
                                        opt.ngf, opt.which_model_netG, opt.norm, not opt.no_dropout, self.gpu_ids, skip=skip, opt=opt)


        if not self.isTrain or opt.continue_train:
            which_epoch = opt.which_epoch
            self.load_network(self.netAtt, 'netAtt', which_epoch)
            self.load_network(self.netG_A, 'G_A', which_epoch)

        if self.isTrain:
            self.old_lr = opt.lr
            # self.fake_A_pool = ImagePool(opt.pool_size)
            self.fake_B_pool = ImagePool(opt.pool_size)
            # define loss functions

            if opt.use_wgan:
                self.criterionGAN = networks.DiscLossWGANGP()
            else:
                self.criterionGAN = networks.GANLoss(use_lsgan=not opt.no_lsgan, tensor=self.Tensor)
            if opt.use_mse:
                self.criterionCycle = torch.nn.MSELoss()
            else:
                self.criterionCycle = torch.nn.L1Loss()
            self.criterionL1 = torch.nn.L1Loss()
            self.criterionIdt = torch.nn.L1Loss()
            # initialize optimizers
            self.optimizer_aU = torch.optim.Adam(self.netAtt.parameters(), lr=1e-4)
            self.optimizer_G = torch.optim.Adam(self.netG_A.parameters(),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))

        print('---------- Networks initialized -------------')
        networks.print_network(self.netAtt)
        networks.print_network(self.netG_A)
        if opt.isTrain:
            self.netAtt.train()
            self.netG_A.train()
        else:
            self.netG_A.eval()
        print('-----------------------------------------------')

    def set_input(self, input):
        input_rgblow = input[0]['A']
        input_rgbnorm = input[0]['B']
        input_attlow = input[1]['C']
        input_attnorm = input[1]['D']
        self.input_rgblow.resize_(input_rgblow.size()).copy_(input_rgblow)
        self.input_rgbnorm.resize_(input_rgbnorm.size()).copy_(input_rgbnorm)
        self.input_attlow.resize_(input_attlow.size()).copy_(input_attlow)
        self.input_attnorm.resize_(input_attnorm.size()).copy_(input_attnorm)
        self.image_paths = input[0]['A_paths']

    def set_input1(self, input):
        input_rgblow = input['A']
        input_rgbnorm = input['B']
        self.input_rgblow.resize_(input_rgblow.size()).copy_(input_rgblow)
        self.input_rgbnorm.resize_(input_rgbnorm.size()).copy_(input_rgbnorm)


    def set_input2(self, input):
        input_attlow = input['C']
        input_attnorm = input['D']
        self.input_attlow.resize_(input_attlow.size()).copy_(input_attlow)
        self.input_attnorm.resize_(input_attnorm.size()).copy_(input_attnorm)
    


    def test(self):
        self.real_A = Variable(self.input_A, volatile=True)
        self.real_A_gray = Variable(self.input_A_gray, volatile=True)
        if self.opt.noise > 0:
            self.noise = Variable(torch.cuda.FloatTensor(self.real_A.size()).normal_(mean=0, std=self.opt.noise/255.))
            self.real_A = self.real_A + self.noise
        if self.opt.input_linear:
            self.real_A = (self.real_A - torch.min(self.real_A))/(torch.max(self.real_A) - torch.min(self.real_A))
        if self.opt.skip == 1:
            self.fake_B, self.latent_real_A = self.netG_A.forward(self.real_A, self.real_A_gray)
        else:
            self.fake_B = self.netG_A.forward(self.real_A, self.real_A_gray)

        self.real_B = Variable(self.input_B, volatile=True)


    def predict(self):
        with torch.no_grad():
            self.input_attlowtest = Variable(self.input_attlow)
            self.input_rgblowtest = Variable(self.input_rgblow)
            self.input_attnormtest = Variable(self.input_attnorm)
        self.attgentest = self.netAtt.forward(self.input_attlowtest)
        if self.opt.noise > 0:
            self.noise = Variable(torch.cuda.FloatTensor(self.real_A.size()).normal_(mean=0, std=self.opt.noise/255.))
            self.real_A = self.real_A + self.noise
        if self.opt.input_linear:
            self.real_A = (self.real_A - torch.min(self.real_A))/(torch.max(self.real_A) - torch.min(self.real_A))
        if self.opt.skip == 1:
            # self.rgbgentest = self.netG_A.forward(self.input_rgblowtest, self.input_attnormtest)
            self.rgbgentest = self.netG_A.forward(self.input_rgblowtest, self.attgentest)
        else:
            self.fake_B = self.netG_A.forward(self.real_A, self.real_A_gray)

        input_rgblowtest = util.tensor2im(self.input_rgblowtest.data)
        rgbgentest = util.tensor2im(self.rgbgentest.data)
        attgentest = util.atten2im(self.attgentest.data)
        return OrderedDict([('attgentest', attgentest), ('input_rgblowtest', input_rgblowtest), ('rgbgentest', rgbgentest)])
        # return OrderedDict([('input_rgblowtest', input_rgblowtest), ('rgbgentest', rgbgentest)])


    # get image paths
    def get_image_paths(self):
        return self.image_paths

    def backward_D_basic(self, netD, real, fake, use_ragan):
        # Real
        pred_real = netD.forward(real)
        pred_fake = netD.forward(fake.detach())
        if self.opt.use_wgan:
            loss_D_real = pred_real.mean()
            loss_D_fake = pred_fake.mean()
            loss_D = loss_D_fake - loss_D_real + self.criterionGAN.calc_gradient_penalty(netD, 
                                                real.data, fake.data)
        elif self.opt.use_ragan and use_ragan:
            loss_D = (self.criterionGAN(pred_real - torch.mean(pred_fake), True) +
                                      self.criterionGAN(pred_fake - torch.mean(pred_real), False)) / 2
        else:
            loss_D_real = self.criterionGAN(pred_real, True)
            loss_D_fake = self.criterionGAN(pred_fake, False)
            loss_D = (loss_D_real + loss_D_fake) * 0.5
        # loss_D.backward()
        return loss_D

    def backward_D_A(self):
        fake_B = self.fake_B_pool.query(self.fake_B)
        fake_B = self.fake_B
        self.loss_D_A = self.backward_D_basic(self.netD_A, self.real_B, fake_B, True)
        self.loss_D_A.backward()
    
    def backward_D_P(self):
        if self.opt.hybrid_loss:
            loss_D_P = self.backward_D_basic(self.netD_P, self.real_patch, self.fake_patch, False)
            if self.opt.patchD_3 > 0:
                for i in range(self.opt.patchD_3):
                    loss_D_P += self.backward_D_basic(self.netD_P, self.real_patch_1[i], self.fake_patch_1[i], False)
                self.loss_D_P = loss_D_P/float(self.opt.patchD_3 + 1)
            else:
                self.loss_D_P = loss_D_P
        else:
            loss_D_P = self.backward_D_basic(self.netD_P, self.real_patch, self.fake_patch, True)
            if self.opt.patchD_3 > 0:
                for i in range(self.opt.patchD_3):
                    loss_D_P += self.backward_D_basic(self.netD_P, self.real_patch_1[i], self.fake_patch_1[i], True)
                self.loss_D_P = loss_D_P/float(self.opt.patchD_3 + 1)
            else:
                self.loss_D_P = loss_D_P
        if self.opt.D_P_times2:
            self.loss_D_P = self.loss_D_P*2
        self.loss_D_P.backward()

    def forward(self):
        self.input_rgblow = Variable(self.input_rgblow)
        self.input_rgbnorm = Variable(self.input_rgbnorm)
        self.input_attlow = Variable(self.input_attlow)
        self.input_attnorm = Variable(self.input_attnorm)
        # print('forward!!')
        self.attgen = self.netAtt.forward(self.input_attlow)
        if self.opt.noise > 0:
            self.noise = Variable(torch.cuda.FloatTensor(self.real_A.size()).normal_(mean=0, std=self.opt.noise/255.))
            self.real_A = self.real_A + self.noise
        if self.opt.input_linear:
            self.real_A = (self.real_A - torch.min(self.real_A))/(torch.max(self.real_A) - torch.min(self.real_A))
        if self.opt.skip == 1:
            self.rgbgen = self.netG_A.forward(self.input_rgblow, self.attgen.detach())
            # self.rgbgen = self.netG_A.forward(self.input_rgblow, self.attgen)
        else:
            self.fake_B = self.netG_A.forward(self.real_img, self.real_A_gray)
        if self.opt.patchD:
            w = self.input_rgblow.size(3)
            h = self.input_rgblow.size(2)
            w_offset = random.randint(0, max(0, w - self.opt.patchSize - 1))
            h_offset = random.randint(0, max(0, h - self.opt.patchSize - 1))

            self.fake_patch = self.rgbgen[:,:, h_offset:h_offset + self.opt.patchSize,
                   w_offset:w_offset + self.opt.patchSize]
            self.real_patch = self.input_rgbnorm[:,:, h_offset:h_offset + self.opt.patchSize,
                   w_offset:w_offset + self.opt.patchSize]
            self.input_patch = self.input_rgblow[:,:, h_offset:h_offset + self.opt.patchSize,
                   w_offset:w_offset + self.opt.patchSize]
        if self.opt.patchD_3 > 0:
            self.fake_patch_1 = []
            self.real_patch_1 = []
            self.input_patch_1 = []
            w = self.input_rgblow.size(3)
            h = self.input_rgblow.size(2)
            for i in range(self.opt.patchD_3):
                w_offset_1 = random.randint(0, max(0, w - self.opt.patchSize - 1))
                h_offset_1 = random.randint(0, max(0, h - self.opt.patchSize - 1))
                self.fake_patch_1.append(self.rgbgen[:,:, h_offset_1:h_offset_1 + self.opt.patchSize,
                    w_offset_1:w_offset_1 + self.opt.patchSize])
                self.real_patch_1.append(self.input_rgbnorm[:,:, h_offset_1:h_offset_1 + self.opt.patchSize,
                    w_offset_1:w_offset_1 + self.opt.patchSize])
                self.input_patch_1.append(self.input_rgblow[:,:, h_offset_1:h_offset_1 + self.opt.patchSize,
                    w_offset_1:w_offset_1 + self.opt.patchSize])

    def backward_aU(self, epoch):
        self.loss_attUnet = networks.MSELoss(self.attgen, self.input_attnorm)
        self.loss_attUnet.backward()

    def backward_G(self, epoch):
        # print('backward')
        # self.loss_attUnet = networks.MSELoss(self.attgen, self.input_attnorm)
        self.loss_rgbUnet1 = networks.MAELoss(self.rgbgen, self.input_rgbnorm)
        self.loss_rgbUnet2 = networks.MS_SSIMLoss(self.rgbgen, self.input_rgbnorm)
        # print(self.loss_attUnet, self.loss_rgbUnet1, self.loss_rgbUnet2)

        if epoch < 0:
            vgg_w = 0
        else:
            vgg_w = 1
        if self.opt.vgg > 0:
            self.loss_vgg_b = self.vgg_loss.compute_vgg_loss(self.vgg, 
                    self.rgbgen, self.input_rgbnorm) * self.opt.vgg if self.opt.vgg > 0 else 0
            if self.opt.patch_vgg:
                if not self.opt.IN_vgg:
                    loss_vgg_patch = self.vgg_loss.compute_vgg_loss(self.vgg, 
                    self.fake_patch, self.real_patch) * self.opt.vgg
                else:
                    loss_vgg_patch = self.vgg_patch_loss.compute_vgg_loss(self.vgg, 
                    self.fake_patch, self.real_patch) * self.opt.vgg
                if self.opt.patchD_3 > 0:
                    for i in range(self.opt.patchD_3):
                        if not self.opt.IN_vgg:
                            loss_vgg_patch += self.vgg_loss.compute_vgg_loss(self.vgg, 
                                self.fake_patch_1[i], self.real_patch_1[i]) * self.opt.vgg
                        else:
                            loss_vgg_patch += self.vgg_patch_loss.compute_vgg_loss(self.vgg, 
                                self.fake_patch_1[i], self.real_patch_1[i]) * self.opt.vgg
                    self.loss_vgg_b += loss_vgg_patch/float(self.opt.patchD_3 + 1)
                else:
                    self.loss_vgg_b += loss_vgg_patch
            # self.loss_tol = 10 * self.loss_attUnet + 0.1 * self.loss_rgbUnet1 + 0.65 * self.loss_rgbUnet2 + 0.25 * self.loss_vgg_b*vgg_w
            self.loss_tol = 0.15 * self.loss_rgbUnet1 +  0.85 * self.loss_rgbUnet2 + self.loss_vgg_b*vgg_w
        elif self.opt.fcn > 0:
            self.loss_fcn_b = self.fcn_loss.compute_fcn_loss(self.fcn, 
                    self.fake_B, self.real_A) * self.opt.fcn if self.opt.fcn > 0 else 0
            if self.opt.patchD:
                loss_fcn_patch = self.fcn_loss.compute_vgg_loss(self.fcn, 
                    self.fake_patch, self.input_patch) * self.opt.fcn
                if self.opt.patchD_3 > 0:
                    for i in range(self.opt.patchD_3):
                        loss_fcn_patch += self.fcn_loss.compute_vgg_loss(self.fcn, 
                            self.fake_patch_1[i], self.input_patch_1[i]) * self.opt.fcn
                    self.loss_fcn_b += loss_fcn_patch/float(self.opt.patchD_3 + 1)
                else:
                    self.loss_fcn_b += loss_fcn_patch
            self.loss_G = self.loss_G_A + self.loss_fcn_b*vgg_w
        # print(self.loss_tol)
        self.loss_tol.backward()

    def optimize_parameters(self, epoch):
        # divide backward
        # forward
        self.forward()
        # G_A and G_B
        self.optimizer_aU.zero_grad()
        self.backward_aU(epoch)
        # self.loss_attUnet.backward()
        self.optimizer_aU.step()

        self.optimizer_G.zero_grad()
        # self.loss_tol.backward()
        self.backward_G(epoch)
        self.optimizer_G.step()


    def get_current_errors(self, epoch):
        aU = self.loss_attUnet.item()
        G = self.loss_tol.item()
        if self.opt.vgg > 0:
            vgg = self.loss_vgg_b.item()/self.opt.vgg if self.opt.vgg > 0 else 0
            return OrderedDict([('aU', aU), ('G', G), ("vgg", vgg)])
            # return OrderedDict([('G', G), ("vgg", vgg)])
        elif self.opt.fcn > 0:
            fcn = self.loss_fcn_b.item()/self.opt.fcn if self.opt.fcn > 0 else 0
            return OrderedDict([('D_A', D_A), ('G_A', G_A), ("fcn", fcn), ("D_P", D_P)])
        

    def get_current_visuals(self):
        attlow = util.atten2im(self.input_attlow.data)
        rgblow = util.tensor2im(self.input_rgblow.data)
        if self.opt.skip > 0:

            rgbgen = util.tensor2im(self.rgbgen.data)
            # rgbgen_show = util.latent2im(self.rgbgen.data)
            if self.opt.patchD:
                fake_patch = util.tensor2im(self.fake_patch.data)
                real_patch = util.tensor2im(self.real_patch.data)
                if self.opt.patch_vgg:
                    input_patch = util.tensor2im(self.input_patch.data)
                    if not self.opt.self_attention:
                        return OrderedDict([('real_A', real_A), ('fake_B', fake_B), ('latent_real_A', latent_real_A),
                                ('latent_show', latent_show), ('real_B', real_B), ('real_patch', real_patch),
                                ('fake_patch', fake_patch), ('input_patch', input_patch)])
                    else:
                        attgen = util.atten2im(self.attgen.data)
                        return OrderedDict([('attlow', attlow), ('attgen', attgen), ('rgblowA', rgblow),
                                            ('rgbgen', rgbgen)])
                        # return OrderedDict([('attlow', attlow), ('attgen', attgen), ('rgblowA', rgblow),
                        #         ('rgbgen', rgbgen), ('rgbgen_show', rgbgen_show)])
                        # return OrderedDict([('attlow', attlow), ('rgblowA', rgblow),
                        #                     ('rgbgen', rgbgen), ('rgbgen_show', rgbgen_show)])
                        # return OrderedDict([('attlow', attlow), ('rgblowA', rgblow),
                        #                     ('rgbgen', rgbgen)])
                else:
                    if not self.opt.self_attention:
                        return OrderedDict([('real_A', real_A), ('fake_B', fake_B), ('latent_real_A', latent_real_A),
                                ('latent_show', latent_show), ('real_B', real_B), ('real_patch', real_patch),
                                ('fake_patch', fake_patch)])
                    else:
                        self_attention = util.atten2im(self.real_A_gray.data)
                        return OrderedDict([('real_A', real_A), ('fake_B', fake_B), ('latent_real_A', latent_real_A),
                                ('latent_show', latent_show), ('real_B', real_B), ('real_patch', real_patch),
                                ('fake_patch', fake_patch), ('self_attention', self_attention)])
            else:
                if not self.opt.self_attention:
                    return OrderedDict([('attlow', attlow), ('attgen', attgen), ('rgblowA', rgblow),
                                        ('rgbgen', rgbgen)])
                    # return OrderedDict([('attlow', attlow), ('attgen', attgen), ('rgblowA', rgblow),
                    #             ('rgbgen', rgbgen), ('rgbgen_show', rgbgen_show)])
                else:
                    self_attention = util.atten2im(self.real_A_gray.data)
                    return OrderedDict([('real_A', real_A), ('fake_B', fake_B), ('real_B', real_B),
                                    ('latent_real_A', latent_real_A), ('latent_show', latent_show),
                                    ('self_attention', self_attention)])
        else:
            if not self.opt.self_attention:
                return OrderedDict([('real_A', real_A), ('fake_B', fake_B), ('real_B', real_B)])
            else:
                self_attention = util.atten2im(self.real_A_gray.data)
                return OrderedDict([('real_A', real_A), ('fake_B', fake_B), ('real_B', real_B),
                                    ('self_attention', self_attention)])

    def save(self, label):
        self.save_network(self.netAtt, 'netAtt', label, self.gpu_ids)
        self.save_network(self.netG_A, 'G_A', label, self.gpu_ids)

    def update_learning_rate(self):
        
        if self.opt.new_lr:
            lr = self.old_lr/2
        else:
            lrd = self.opt.lr / self.opt.niter_decay
            lr = self.old_lr - lrd
        for param_group in self.optimizer_G.param_groups:
            param_group['lr'] = lr

        print('update learning rate: %f -> %f' % (self.old_lr, lr))
        self.old_lr = lr
