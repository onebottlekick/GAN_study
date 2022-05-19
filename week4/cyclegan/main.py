import itertools
import time
import datetime
import sys

import torch
from torch import nn
from torchvision.utils import save_image, make_grid

from config import *
from model import *
from utils import *


criterion_gan = nn.MSELoss()
criterion_cycle = nn.L1Loss()
criterion_identity = nn.L1Loss()

G_AB = Generator(input_shape, n_residual_blocks).to(device)
G_BA = Generator(input_shape, n_residual_blocks).to(device)
D_A = Discriminator(input_shape).to(device)
D_B = Discriminator(input_shape).to(device)

if epoch != 0:
    G_AB.load_state_dict(torch.load("models/%s/G_AB_%d.pth" % (dataset_name, epoch)))
    G_BA.load_state_dict(torch.load("models/%s/G_BA_%d.pth" % (dataset_name, epoch)))
    D_A.load_state_dict(torch.load("models/%s/D_A_%d.pth" % (dataset_name, epoch)))
    D_B.load_state_dict(torch.load("models/%s/D_B_%d.pth" % (dataset_name, epoch)))
else:
    G_AB.apply(weights_init_normal)
    G_BA.apply(weights_init_normal)
    D_A.apply(weights_init_normal)
    D_B.apply(weights_init_normal)
    
optimizer_G = torch.optim.Adam(itertools.chain(G_AB.parameters(), G_BA.parameters()), lr=lr, betas=(b1, b2))
optimizer_D_A = torch.optim.Adam(D_A.parameters(), lr=lr, betas=(b1, b2))
optimizer_D_B = torch.optim.Adam(D_B.parameters(), lr=lr, betas=(b1, b2))

lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(optimizer_G, lr_lambda=LambdaLR(n_epochs, epoch, decay_epoch).step)
lr_scheduler_D_A = torch.optim.lr_scheduler.LambdaLR(optimizer_D_A, lr_lambda=LambdaLR(n_epochs, epoch, decay_epoch).step)
lr_scheduler_D_B = torch.optim.lr_scheduler.LambdaLR(optimizer_D_B, lr_lambda=LambdaLR(n_epochs, epoch, decay_epoch).step)

fake_A_buffer = ReplayBuffer()
fake_B_buffer = ReplayBuffer()

def sample_images(batches_done):
    imgs = next(iter(val_dataloader))
    G_AB.eval()
    G_BA.eval()
    real_A = torch.tensor(imgs["A"], device=device)
    fake_B = G_AB(real_A)
    real_B = torch.tensor(imgs["B"], device=device)
    fake_A = G_BA(real_B)

    real_A = make_grid(real_A, nrow=5, normalize=True)
    real_B = make_grid(real_B, nrow=5, normalize=True)
    fake_A = make_grid(fake_A, nrow=5, normalize=True)
    fake_B = make_grid(fake_B, nrow=5, normalize=True)

    image_grid = torch.cat((real_A, fake_B, real_B, fake_A), 1)
    save_image(image_grid, f"images/{dataset_name}/{batches_done}.png", normalize=False)
    
prev_time = time.time()
for epoch in range(epoch, n_epochs):
    for i, batch in enumerate(train_dataloader):
        real_A = torch.tensor(batch['A'], device=device)
        real_B = torch.tensor(batch['B'], device=device)
        
        valid = torch.ones(real_A.shape[0], *D_A.output_shape, requires_grad=False, device=device)
        fake = torch.zeros(real_A.shape[0], *D_A.output_shape, requires_grad=False, device=device)

        G_AB.train()
        G_BA.train()

        optimizer_G.zero_grad()
        
        loss_id_A = criterion_identity(G_BA(real_A), real_A)
        loss_id_B = criterion_identity(G_AB(real_B), real_B)

        loss_identity = (loss_id_A + loss_id_B) / 2

        fake_B = G_AB(real_A)
        loss_GAN_AB = criterion_gan(D_B(fake_B), valid)
        fake_A = G_BA(real_B)
        loss_GAN_BA = criterion_gan(D_A(fake_A), valid)

        loss_GAN = (loss_GAN_AB + loss_GAN_BA) / 2

        recov_A = G_BA(fake_B)
        loss_cycle_A = criterion_cycle(recov_A, real_A)
        recov_B = G_AB(fake_A)
        loss_cycle_B = criterion_cycle(recov_B, real_B)

        loss_cycle = (loss_cycle_A + loss_cycle_B) / 2

        loss_G = loss_GAN + lambda_cyc * loss_cycle + lambda_id * loss_identity

        loss_G.backward()
        optimizer_G.step()
        
        optimizer_D_A.zero_grad()

        loss_real = criterion_gan(D_A(real_A), valid)
        fake_A_ = fake_A_buffer.push_and_pop(fake_A)
        loss_fake = criterion_gan(D_A(fake_A_.detach()), fake)
        loss_D_A = (loss_real + loss_fake) / 2

        loss_D_A.backward()
        optimizer_D_A.step()
        
        optimizer_D_B.zero_grad()

        loss_real = criterion_gan(D_B(real_B), valid)
        fake_B_ = fake_B_buffer.push_and_pop(fake_B)
        loss_fake = criterion_gan(D_B(fake_B_.detach()), fake)
        loss_D_B = (loss_real + loss_fake) / 2

        loss_D_B.backward()
        optimizer_D_B.step()

        loss_D = (loss_D_A + loss_D_B) / 2
        
        batches_done = epoch * len(train_dataloader) + i
        batches_left = n_epochs * len(train_dataloader) - batches_done
        time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
        prev_time = time.time()
        
        sys.stdout.write(
            "\r[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f, adv: %f, cycle: %f, identity: %f] ETA: %s"
            % (
                epoch,
                n_epochs,
                i,
                len(train_dataloader),
                loss_D.item(),
                loss_G.item(),
                loss_GAN.item(),
                loss_cycle.item(),
                loss_identity.item(),
                time_left,
            )
        )

        if batches_done % sample_interval == 0:
            sample_images(batches_done)
            
    lr_scheduler_G.step()
    lr_scheduler_D_A.step()
    lr_scheduler_D_B.step()
    
    if checkpoint_interval != -1 and epoch % checkpoint_interval == 0:
        torch.save(G_AB.state_dict(), "models/%s/G_AB_%d.pth" % (dataset_name, epoch))
        torch.save(G_BA.state_dict(), "models/%s/G_BA_%d.pth" % (dataset_name, epoch))
        torch.save(D_A.state_dict(), "models/%s/D_A_%d.pth" % (dataset_name, epoch))
        torch.save(D_B.state_dict(), "models/%s/D_B_%d.pth" % (dataset_name, epoch))