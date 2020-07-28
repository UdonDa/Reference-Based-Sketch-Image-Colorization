from model import Generator, Discriminator
from torch.autograd import Variable
from torchvision.utils import save_image
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import time
import datetime
from sys import exit

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
    elif classname.find('InstanceNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

class Solver(object):

    def __init__(self, data_loader, config):
        """Initialize configurations."""

        self.data_loader = data_loader
        self.config = config
        
        self.build_model(config)
        
    def build_model(self, config):
        """Create a generator and a discriminator."""
        self.G = Generator(config.g_conv_dim, config.d_channel, config.channel_1x1)   # 2 for mask vector.
        self.D = Discriminator(config.crop_size, config.d_conv_dim, config.d_repeat_num)
        
        self.G.apply(weights_init)
        self.D.apply(weights_init)

        self.G.cuda()
        self.D.cuda()
        
        self.g_optimizer = torch.optim.Adam(self.G.parameters(), config.g_lr, [config.beta1, config.beta2])
        self.d_optimizer = torch.optim.Adam(self.D.parameters(), config.d_lr, [config.beta1, config.beta2])
        
        self.G = nn.DataParallel(self.G)
        self.D = nn.DataParallel(self.D)
        

    def restore_model(self, resume_iters):
        """Restore the trained generator and discriminator."""
        print('Loading the trained models from step {}...'.format(resume_iters))
        G_path = os.path.join(self.model_save_dir, '{}-G.ckpt'.format(resume_iters))
        D_path = os.path.join(self.model_save_dir, '{}-D.ckpt'.format(resume_iters))
        self.G.load_state_dict(torch.load(G_path, map_location=lambda storage, loc: storage))
        self.D.load_state_dict(torch.load(D_path, map_location=lambda storage, loc: storage))

    def reset_grad(self):
        """Reset the gradient buffers."""
        self.g_optimizer.zero_grad()
        self.d_optimizer.zero_grad()
        
    def update_lr(self, g_lr, d_lr):
        """Decay learning rates of the generator and discriminator."""
        for param_group in self.g_optimizer.param_groups:
            param_group['lr'] = g_lr
        for param_group in self.d_optimizer.param_groups:
            param_group['lr'] = d_lr

    def denorm(self, x):
        """Convert the range from [-1, 1] to [0, 1]."""
        out = (x + 1) / 2
        return out.clamp_(0, 1)

    def gradient_penalty(self, y, x):
        """Compute gradient penalty: (L2_norm(dy/dx) - 1)**2."""
        weight = torch.ones(y.size()).cuda()
        dydx = torch.autograd.grad(outputs=y,
                                   inputs=x,
                                   grad_outputs=weight,
                                   retain_graph=True,
                                   create_graph=True,
                                   only_inputs=True)[0]

        dydx = dydx.view(dydx.size(0), -1)
        dydx_l2norm = torch.sqrt(torch.sum(dydx**2, dim=1))
        return torch.mean((dydx_l2norm-1)**2)

    def train(self):
        data_loader = self.data_loader
        config = self.config

        # Learning rate cache for decaying.
        g_lr = config.g_lr
        d_lr = config.d_lr
        
        # Label for lsgan
        real_target = torch.full((config.batch_size,), 1.).cuda()
        fake_target = torch.full((config.batch_size,), 0.).cuda()
        criterion = nn.MSELoss().cuda()

        # Start training.
        print('Start training...')
        start_time = time.time()
        iteration = 0
        num_iters_decay = config.num_epoch_decay * len(data_loader)
        for epoch in range(config.num_epoch):
            
            for i, (I_ori, I_gt, I_r, I_s) in enumerate(data_loader):
                iteration += i
                
                I_ori = I_ori.cuda(non_blocking=True)
                I_gt = I_gt.cuda(non_blocking=True)
                I_r = I_r.cuda(non_blocking=True)
                I_s = I_s.cuda(non_blocking=True)                

                # =================================================================================== #
                #                             2. Train the discriminator                              #
                # =================================================================================== #

                # Compute loss with real images.
                out = self.D(I_gt)
                d_loss_real = criterion(out, real_target) * 0.5

                # Compute loss with fake images.
                I_fake = self.G(I_r, I_s)
                out = self.D(I_fake.detach())
                d_loss_fake = criterion(out, fake_target) * 0.5

                # Backward and optimize.
                d_loss = d_loss_real + d_loss_fake
                self.reset_grad()
                d_loss.backward()
                self.d_optimizer.step()

                # Logging.
                loss = {}
                loss['D/loss_real'] = d_loss_real.item()
                loss['D/loss_fake'] = d_loss_fake.item()
                
                # =================================================================================== #
                #                               3. Train the generator                                #
                # =================================================================================== #
                
                # if (i+1) % config.n_critic == 0:
                I_fake = self.G(I_r, I_s)
                out = self.D(I_fake)
                g_loss_fake = criterion(out, real_target)
                
                g_loss_rec = torch.mean(torch.abs(I_fake - I_gt)) # Eq.(6)

                # Backward and optimize.
                g_loss = g_loss_fake + config.lambda_rec * g_loss_rec
                self.reset_grad()
                g_loss.backward()
                self.g_optimizer.step()

                # Logging.
                loss['G/loss_fake'] = g_loss_fake.item()
                loss['G/loss_rec'] = g_loss_rec.item()
                    
            # =================================================================================== #
            #                                 4. Miscellaneous                                    #
            # =================================================================================== #

                # Print out training information.
                if (i+1) % config.log_step == 0:
                    et = time.time() - start_time
                    et = str(datetime.timedelta(seconds=et))[:-7]
                    log = "Elapsed [{}], Epoch [{}/{}], Iteration [{}/{}], g_lr {:.5f}, d_lr {:.5f}".format(
                        et, epoch, config.num_epoch, i+1, len(data_loader),
                        g_lr, d_lr)
                    for tag, value in loss.items():
                        log += ", {}: {:.4f}".format(tag, value)
                    print(log)
                    
                # Decay learning rates.
                if (epoch+1) > config.num_epoch_decay:
                    g_lr -= (config.g_lr / float(num_iters_decay))
                    d_lr -= (config.d_lr / float(num_iters_decay))
                    self.update_lr(g_lr, d_lr)
                    # print ('Decayed learning rates, g_lr: {}, d_lr: {}.'.format(g_lr, d_lr))


            # Translate fixed images for debugging.
            if (epoch+1) % config.sample_epoch == 0:
                with torch.no_grad():
                    sample_path = os.path.join(config.sample_dir, '{}.jpg'.format(epoch))
                    I_concat = self.denorm(torch.cat([I_ori, I_gt, I_r, I_fake], dim=2))
                    I_concat = torch.cat([I_concat, I_s.repeat(1,3,1,1)], dim=2)
                    save_image(I_concat.data.cpu(), sample_path)
                    print('Saved real and fake images into {}...'.format(sample_path))

            # Save model checkpoints.
            if (epoch+1) % config.model_save_step == 0:
                G_path = os.path.join(config.model_save_dir, '{}-G.ckpt'.format(epoch+1))
                D_path = os.path.join(config.model_save_dir, '{}-D.ckpt'.format(epoch+1))
                torch.save(self.G.state_dict(), G_path)
                torch.save(self.D.state_dict(), D_path)
                print('Saved model checkpoints into {}...'.format(config.model_save_dir))
            

    def test(self):
        """Translate images using StarGAN trained on a single dataset."""
        # Load the trained generator.
        self.restore_model(self.test_iters)
        
        # Set data loader.
        if self.dataset == 'CelebA':
            data_loader = self.celeba_loader
        elif self.dataset == 'RaFD':
            data_loader = self.rafd_loader
        
        with torch.no_grad():
            for i, (x_real, c_org) in enumerate(data_loader):

                # Prepare input images and target domain labels.
                x_real = x_real.to(self.device)
                c_trg_list = self.create_labels(c_org, self.c_dim, self.dataset, self.selected_attrs)

                # Translate images.
                x_fake_list = [x_real]
                for c_trg in c_trg_list:
                    x_fake_list.append(self.G(x_real, c_trg))

                # Save the translated images.
                x_concat = torch.cat(x_fake_list, dim=3)
                result_path = os.path.join(self.result_dir, '{}-images.jpg'.format(i+1))
                save_image(self.denorm(x_concat.data.cpu()), result_path, nrow=1, padding=0)
                print('Saved real and fake images into {}...'.format(result_path))
