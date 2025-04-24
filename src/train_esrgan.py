import os
import argparse
from tqdm import tqdm

import torch
import torch.nn as nn
import torchvision.utils as vutils
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim

from esrgan_model import GeneratorRRDB, Discriminator, VGGFeatureExtractor
from data_loader import get_dataloaders


def compute_metrics(sr, hr):
    """Center‐crop HR to SR size and compute PSNR & SSIM."""
    sr_np = sr.permute(0,2,3,1).cpu().numpy()
    hr_np = hr.permute(0,2,3,1).cpu().numpy()
    psnr_vals, ssim_vals = [], []

    for i in range(sr_np.shape[0]):
        sr_img = sr_np[i]
        hr_img = hr_np[i]
        h, w = sr_img.shape[:2]
        # center‐crop HR
        hr_img = hr_img[
            (hr_img.shape[0]-h)//2:(hr_img.shape[0]+h)//2,
            (hr_img.shape[1]-w)//2:(hr_img.shape[1]+w)//2
        ]
        psnr_vals.append(psnr(hr_img, sr_img, data_range=1.0))
        ssim_vals.append(ssim(hr_img, sr_img, channel_axis=-1, data_range=1.0))
    return sum(psnr_vals)/len(psnr_vals), sum(ssim_vals)/len(ssim_vals)


def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Data loaders
    train_loader, val_loader = get_dataloaders(
        args.train_lr, args.train_hr,
        args.val_lr, args.val_hr,
        hr_crop_size=args.hr_crop,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )

    # Models
    netG = GeneratorRRDB().to(device)
    netD = Discriminator().to(device)
    vgg = VGGFeatureExtractor(device=device).to(device)
    for p in vgg.parameters(): p.requires_grad = False

    # Losses
    adversarial_criterion = nn.BCEWithLogitsLoss().to(device)
    content_criterion     = nn.MSELoss().to(device)

    # Optimizers
    optimizerG = torch.optim.Adam(netG.parameters(), lr=args.lr, betas=(0.9,0.999))
    optimizerD = torch.optim.Adam(netD.parameters(), lr=args.lr, betas=(0.9,0.999))

    best_psnr = 0.0
    for epoch in range(1, args.epochs+1):
        netG.train(); netD.train()
        loop = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")

        for lr, hr in loop:
            lr, hr = lr.to(device), hr.to(device)
            batch_size = lr.size(0)

            # Real / Fake labels
            valid = torch.ones((batch_size,1), device=device)
            fake  = torch.zeros((batch_size,1), device=device)

            # ------------------
            #  Train Discriminator (Relativistic)
            # ------------------
            optimizerD.zero_grad()
            sr = netG(lr).detach()
            pred_real = netD(hr)
            pred_fake = netD(sr)

            # RaGAN discriminator loss
            real_loss = adversarial_criterion(pred_real - pred_fake.mean(), valid)
            fake_loss = adversarial_criterion(pred_fake - pred_real.mean().detach(), fake)
            d_loss = (real_loss + fake_loss) / 2
            d_loss.backward()
            optimizerD.step()

            # ------------------
            #  Train Generator
            # ------------------
            optimizerG.zero_grad()
            sr = netG(lr)
            pred_real = netD(hr).detach()
            pred_fake = netD(sr)

            # Relativistic adversarial loss for G
            g_adv = adversarial_criterion(pred_fake - pred_real.mean(), valid)

            # Perceptual (content) loss before activation
            feat_sr = vgg(sr)
            feat_hr = vgg(hr)
            # if shapes mismatch, interpolate to match
            if feat_sr.shape != feat_hr.shape:
                _,_,h,w = feat_hr.shape
                feat_sr = nn.functional.interpolate(feat_sr, size=(h,w), mode='bilinear', align_corners=False)
            content_loss = content_criterion(feat_sr, feat_hr)

            # Total generator loss
            g_loss = content_loss + args.adv_weight * g_adv
            g_loss.backward()
            optimizerG.step()

            loop.set_postfix(D_loss=d_loss.item(), G_loss=g_loss.item())

        # ------------------
        #  Validation
        # ------------------
        netG.eval()
        running_psnr, running_ssim = 0.0, 0.0
        with torch.no_grad():
            for lr, hr in val_loader:
                lr, hr = lr.to(device), hr.to(device)
                sr = netG(lr)
                p, s = compute_metrics(sr, hr)
                running_psnr += p
                running_ssim += s
        val_psnr = running_psnr / len(val_loader)
        val_ssim = running_ssim / len(val_loader)
        print(f"Validation → PSNR: {val_psnr:.4f} dB, SSIM: {val_ssim:.4f}")

        # Checkpoint & sample save
        if val_psnr > best_psnr:
            best_psnr = val_psnr
            torch.save(netG.state_dict(), os.path.join(args.checkpoint_dir, 'generator_best.pth'))
            print("🚀 Saved best generator checkpoint.")
        # Save sample grid
        sample_lr, sample_hr = next(iter(val_loader))
        sample_sr = netG(sample_lr.to(device))
        grid = vutils.make_grid(
            torch.cat([sample_lr, sample_sr.cpu(), sample_hr], dim=0),
            nrow=args.batch_size
        )
        vutils.save_image(grid, os.path.join(args.sample_dir, f'epoch_{epoch}.png'), normalize=True)

    print("Training complete.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train ESRGAN (×2) on DIV2K")
    parser.add_argument('--train-lr',     type=str, required=True)
    parser.add_argument('--train-hr',     type=str, required=True)
    parser.add_argument('--val-lr',       type=str, required=True)
    parser.add_argument('--val-hr',       type=str, required=True)
    parser.add_argument('--epochs',       type=int,   default=100)
    parser.add_argument('--batch-size',   type=int,   default=16)
    parser.add_argument('--lr',           type=float, default=1e-4)
    parser.add_argument('--adv-weight',   type=float, default=5e-3,
                        help="Weight for adversarial loss (recommend 0.005)")
    parser.add_argument('--hr-crop',      type=int,   default=96)
    parser.add_argument('--num-workers',  type=int,   default=4)
    parser.add_argument('--checkpoint-dir',type=str,  default='checkpoints_esrgan')
    parser.add_argument('--sample-dir',   type=str,   default='samples_esrgan')
    args = parser.parse_args()

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.sample_dir, exist_ok=True)
    train(args)
