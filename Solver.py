import os
from time import gmtime, strftime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from model import (CLUBSample, Classifier, Discriminator, Disentanglement, Encoder,
                   Generator, Projection, Reconstruction)
from utils import *


class Solver():
    def __init__(self, args, dir = './'):

        timestring = strftime("%Y-%m-%d_%H-%M-%S", gmtime()) + "_%s" % args.Task
        self.logdir = os.path.join(dir, 'logs', timestring)
        self.logger = SummaryWriter(log_dir=self.logdir)
        self.device = torch.device("cuda")

        self.batch_size = args.Batch_size
        self.lr = args.learning_rate
        self.mi_coef = args.mi_coef
        self.mi_iter = args.mi_iter
        self.para = args.para

        self.E = Encoder()
        self.C = Classifier()
        self.D = Disentanglement()
        self.Proj0 = Projection()
        self.Proj1 = Projection()
        self.R = Reconstruction()
        
        self.Gen = Generator()
        self.Dis = Discriminator()
        self.MI  = CLUBSample()

        self.modules = nn.ModuleDict({
            'E':    self.E,
            'C':    self.C,
            'D':    self.D,
            'Proj0':self.Proj0,
            'Proj1':self.Proj1,
            'R':    self.R,
            'Gen':  self.Gen,
            'Dis':  self.Dis,
            'MI':   self.MI,
        })
        
        self.xent_loss = nn.CrossEntropyLoss()
        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCELoss()
        
        self.set_optimizer()
        self.to_device()

    def to_device(self):
        for k, v in self.modules.items():
            self.modules[k] = v.to(self.device)

    def set_optimizer(self):
        self.opt = {
            'E':    optim.Adam(self.E.parameters(),     lr=self.lr, weight_decay=1e-5, amsgrad=True),
            'C':    optim.Adam(self.C.parameters(),     lr=self.lr, weight_decay=1e-5, amsgrad=True),
            'D':    optim.Adam(self.D.parameters(),     lr=self.lr, weight_decay=1e-5, amsgrad=True),
            'Proj0':optim.Adam(self.Proj0.parameters(), lr=self.lr, weight_decay=1e-5, amsgrad=True),
            'Proj1':optim.Adam(self.Proj1.parameters(), lr=self.lr, weight_decay=1e-5, amsgrad=True),
            'R':    optim.Adam(self.R.parameters(),     lr=self.lr, weight_decay=1e-5, amsgrad=True),
            'Gen':  optim.Adam(self.Gen.parameters(),   lr=self.lr, weight_decay=1e-5, amsgrad=True),
            'Dis':  optim.Adam(self.Dis.parameters(),   lr=self.lr, weight_decay=1e-5, amsgrad=True),
            'MI':   optim.Adam(self.MI.parameters(),    lr=self.lr, weight_decay=1e-5, amsgrad=True),
        }

    def reset_grad(self):
        for _, opt in self.opt.items():
            opt.zero_grad()

    def group_opt_step(self, opt_keys):
        for k in opt_keys:
            torch.nn.utils.clip_grad_norm_(self.modules[k].parameters(), 1.)
            self.opt[k].step()
        self.reset_grad()
    
    def contrastive_loss(self, N, P, proj):
        
        return -(F.cosine_similarity(proj(N), P.detach()) + F.cosine_similarity(N.detach(), proj(P))).mean()
    
    def train_epoch(self, dataset, epoch, prefix = 'ADNC_train'):
        # set training
        for k in self.modules.keys():
            self.modules[k].train()

        # torch.cuda.manual_seed(1)
        total_batches = len(dataset)
        pbar_descr_prefix = "Epoch %d" % (epoch)
        contrastive_loss    = 0.0
        cls_loss            = 0.0
        mi_loss             = 0.0
        mi                  = 0.0
        d_loss              = 0.0
        g_loss              = 0.0
        rec_loss            = 0.0
        
        with tqdm(total=total_batches, ncols=80, dynamic_ncols=False,
                  desc=pbar_descr_prefix) as pbar:

            for batch_idx, data in enumerate(dataset):
                if batch_idx > total_batches:
                    return batch_idx
                
                # ============================================================ #
                """
                data loading
                """
                image0, image1, Age, Gender, Edu, label, RID = data
                Age = Age.to(self.device)
                Gender = Gender.squeeze(-1).long().to(self.device)
                Edu = Edu.to(self.device)
                label = label.squeeze(-1).long().to(self.device)
                image0 = image0.to(self.device)
                image1 = image1.to(self.device)
                
                bts = image0.shape[0]

                # ============================================================ #
                
                self.reset_grad()
                
                # ============================================================ #
                """
                optimize the encoder, disentanglement and classifier by supervised learning and contrastive learning
                """
                feat0, feat1 = self.E(image0), self.E(image1)
                dis0_cls, dis0_meta = self.D(feat0)
                dis1_cls, dis1_meta = self.D(feat1)
                feat0_pre, feat1_pre = self.C(dis0_cls), self.C(dis1_cls)
                
                _loss1 = self.contrastive_loss(dis0_cls, dis1_cls, self.Proj0) + self.contrastive_loss(dis0_meta, dis1_meta, self.Proj1)
                contrastive_loss += _loss1.item()/2

                _loss2 = self.para* (self.xent_loss(feat0_pre, label) + self.xent_loss(feat1_pre, label))
                cls_loss += _loss2.item()
                (_loss1 + _loss2).backward()
                self.group_opt_step(['E', 'D', 'C', 'Proj0', 'Proj1'])

                # ============================================================ #
                """
                optimize the mutual information and disentanglement by CLUBSample
                """ 
                
                dis0_cls, dis0_meta = self.D(feat0.detach())
                dis1_cls, dis1_meta = self.D(feat1.detach())

                _loss3_mi = 0.5* (self.MI(dis0_cls, dis0_meta) + self.MI(dis1_cls, dis1_meta)) * self.mi_coef
                _loss3_mi.backward()
                mi += _loss3_mi.item()
                self.group_opt_step(['D'])
                
                for _ in range(self.mi_iter):
                    dis0_cls, dis0_meta = self.D(feat0.detach())
                    dis1_cls, dis1_meta = self.D(feat1.detach())
                    _loss3 = (0.5* (self.MI.learning_loss(dis0_cls, dis0_meta) + self.MI.learning_loss(dis1_cls, dis1_meta)) * self.mi_coef)
                    _loss3.backward()
                    mi_loss += _loss3.item()
                    self.group_opt_step(['MI'])

                # ============================================================ #
                noise = torch.randn((bts, 60)).to(self.device)               
                c = np.zeros((bts, 2))
                c[range(bts), Gender.cpu().numpy()] = 1
                discret_code = torch.Tensor(c).to(self.device)
                continus_code = torch.cat([Age, Edu], dim=1)
                    
                """
                optimize infogan
                """
                                
                gan_inputs = torch.cat([noise, discret_code, continus_code], 1).to(self.device) # [(B, 60), (B, 2), (B, 2)]
                label_real = torch.ones(bts, 1).to(self.device)
                label_fake = torch.zeros(bts, 1).to(self.device)
                
                # Dis step
                generated_meta = self.Gen(gan_inputs)
                dis0_cls, dis0_meta = self.D(feat0.detach())
                dis1_cls, dis1_meta = self.D(feat1.detach())
                real0_pre, real1_pre = self.Dis(dis0_meta)[0], self.Dis(dis1_meta)[0]
                fake_pre = self.Dis(generated_meta)[0]
                
                loss_real = 0.5* (self.bce_loss(real0_pre, label_real) + self.bce_loss(real1_pre, label_real))
                loss_fake = self.bce_loss(fake_pre, label_fake)
                
                _loss4 = (loss_real + loss_fake)/2
                _loss4.backward()
                d_loss += _loss4.item()
                self.group_opt_step(['D', 'Dis'])
                
                # Gen step
                
                generated_meta = self.Gen(gan_inputs)
                fake_pre, dist_pre, con_pre = self.Dis(generated_meta)
                loss_gen = self.bce_loss(fake_pre, label_real)
                dis_loss = self.xent_loss(dist_pre, Gender)
                con_loss = self.mse_loss(con_pre, continus_code)
                
                _loss5 = loss_gen + dis_loss + 10 * con_loss # we set the weight of 10 by default
                _loss5.backward()
                g_loss += _loss5.item()
                self.group_opt_step(['Gen', 'Dis'])
                
                # ============================================================ #
                """
                optimize the reconstruction and disentanglement by MSE
                """
                
                dis0_cls, dis0_meta = self.D(feat0.detach())
                dis1_cls, dis1_meta = self.D(feat1.detach())
        
                feat0_rec, feat1_rec = self.R(torch.cat([dis0_cls, dis0_meta], 1)), self.R(torch.cat([dis1_cls, dis1_meta], 1))
                feat0_cross, feat1_cross = self.R(torch.cat([dis0_cls, dis1_meta], 1)), self.R(torch.cat([dis1_cls, dis0_meta], 1))
                
                self_rec_loss0  = self.mse_loss(feat0_rec, feat0.detach())
                self_rec_loss1  = self.mse_loss(feat1_rec, feat1.detach())
                cross_rec_loss0 = self.mse_loss(feat0_cross, feat0.detach())
                cross_rec_loss1 = self.mse_loss(feat1_cross, feat1.detach())
                
                generated_meta = self.Gen(gan_inputs)
                feat0_gan, feat1_gan = self.R(torch.cat([dis0_cls, generated_meta], 1)), self.R(torch.cat([dis1_cls, generated_meta], 1))
                gan_rec_loss0   = self.mse_loss(feat0_gan, feat0.detach())
                gan_rec_loss1   = self.mse_loss(feat1_gan, feat1.detach())
            
                _loss6 = (self_rec_loss0 + self_rec_loss1 + cross_rec_loss0 + cross_rec_loss1 + gan_rec_loss0 + gan_rec_loss1) / 6
                _loss6.backward()
                rec_loss += _loss6.item()
                self.group_opt_step(['D', 'R'])
                pbar.update()
            
                # ============================================================ #
        """
        logging information
        """
        self.logger.add_scalar(prefix + "_contrastive_loss"   , contrastive_loss/len(dataset)  , global_step=epoch)
        self.logger.add_scalar(prefix + "_cls_loss"           , cls_loss/len(dataset)  , global_step=epoch)
        self.logger.add_scalar(prefix + "_mi_loss"            , mi_loss/len(dataset)   , global_step=epoch)
        self.logger.add_scalar(prefix + "_mi"                 , mi/len(dataset)        , global_step=epoch)
        self.logger.add_scalar(prefix + "_d_loss"             , d_loss/len(dataset)    , global_step=epoch)
        self.logger.add_scalar(prefix + "_g_loss"             , g_loss/len(dataset)    , global_step=epoch)
        self.logger.add_scalar(prefix + "_rec_loss"           , rec_loss/len(dataset)  , global_step=epoch)
        
        return contrastive_loss, cls_loss, mi_loss, d_loss, g_loss, rec_loss

    def test_epoch(self, dataset, epoch, prefix = 'Val'):
        self.E.eval()
        self.D.eval()
        self.C.eval()
        
        size = 0.0
        cls_loss = 0.0
        correct = 0.0
        pre_list = []
        label_list = []
        with torch.no_grad():
            for batch_idx, data in enumerate(dataset):
                images, _, _, _, label, RID = data
                label = label.to(self.device)
                label = label.squeeze(-1).long().cuda()

                for image in images:
                    feat = self.E(image.to(self.device))
                    dis  = self.D(feat)[0]
                    out = self.C(dis[:, :64])

                    cls_loss += self.xent_loss(out, label).item()
                    pre = out.data.max(1)[1]
                    correct += pre.eq(label.data).cpu().sum()
                    
                    k = label.data.size()[0]
                    size += k
                    
                    pre_list.append(out.detach().cpu().numpy())
                    label_list.append(label.detach().cpu().numpy())
        
        acc = correct / size
        
        self.logger.add_scalar(prefix + "_cls_loss", cls_loss/len(dataset), global_step=epoch)
        self.logger.add_scalar(prefix + "_acc", acc, global_step=epoch)
        
        return cls_loss, acc
