import os
import argparse
from tqdm import tqdm

import torch
import torch.nn as nn
import torchvision.utils as vutils
from torchvision import transforms
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim

from esrgan_model import GeneratorRRDB, Discriminator, VGGFeatureExtractor
from data_loader import get_dataloaders


def compute_metrics(sr, hr):
    sr_np = sr.permute(0, 2, 3, 1).cpu().numpy()
    hr_np = hr.permute(0, 2, 3, 1).cpu().numpy()
    psnr_vals, ssim_vals = [], []

    for i in range(sr_np.shape[0]):
        h, w = sr_np[i].shape[:2]
        hr_crop = hr_np[i][(hr_np[i].shape[0] - h)//2:(hr_np[i].shape[0] + h)//2,
                           (hr_np[i].shape[1] - w)//2:(hr_np[i].shape[1] + w)//2]
        psnr_vals.append(psnr(hr_crop, sr_np[i], data_range=1.0))
        ssim_vals.append(ssim(hr_crop, sr_np[i], data_range=1.0, channel_axis=-1))
    return sum(psnr_vals)/len(psnr_vals), sum(ssim_vals)/len(ssim_vals)


def normalize_vgg(img):
    """Normalize input for VGG"""
    mean = torch.tensor([0.485, 0.456, 0.406], device=img.device).view(1, 3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225], device=img.device).view(1, 3, 1, 1)
    return (img - mean) / std


def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_loader, val_loader = get_dataloaders(
        args.train_lr, args.train_hr,
        args.val_lr, args.val_hr,
        hr_crop_size=args.hr_crop,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )

    netG = GeneratorRRDB().to(device)
    netD = Discriminator().to(device)
    vgg = VGGFeatureExtractor(device=device).to(device)
    for p in vgg.parameters(): p.requires_grad = False

    adversarial_criterion = nn.BCEWithLogitsLoss().to(device)
    content_criterion     = nn.MSELoss().to(device)

    optimizerG = torch.optim.Adam(netG.parameters(), lr=args.lr, betas=(0.9, 0.999))
    optimizerD = torch.optim.Adam(netD.parameters(), lr=args.lr, betas=(0.9, 0.999))

    best_psnr = 0.0
    for epoch in range(1, args.epochs + 1):
        netG.train(); netD.train()
        loop = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")

        for lr, hr in loop:
            lr, hr = lr.to(device), hr.to(device)
            batch_size = lr.size(0)

            valid = torch.ones((batch_size, 1), device=device)
            fake = torch.zeros((batch_size, 1), device=device)

            # Discriminator
            optimizerD.zero_grad()
            sr = netG(lr).detach()
            pred_real = netD(hr)
            pred_fake = netD(sr)
            d_loss = (
                adversarial_criterion(pred_real - pred_fake.mean(), valid) +
                adversarial_criterion(pred_fake - pred_real.mean().detach(), fake)
            ) * 0.5
            d_loss.backward()
            optimizerD.step()

            # Generator
            optimizerG.zero_grad()
            sr = netG(lr)
            pred_real = netD(hr).detach()
            pred_fake = netD(sr)
            g_adv = adversarial_criterion(pred_fake - pred_real.mean(), valid)

            feat_sr = vgg(normalize_vgg(sr))
            feat_hr = vgg(normalize_vgg(hr))
            if feat_sr.shape != feat_hr.shape:
                feat_sr = nn.functional.interpolate(feat_sr, size=feat_hr.shape[-2:], mode='bilinear', align_corners=False)

            content_loss = content_criterion(feat_sr, feat_hr)
            g_loss = content_loss + args.adv_weight * g_adv
            g_loss.backward()
            optimizerG.step()

            loop.set_postfix(D_loss=d_loss.item(), G_loss=g_loss.item())

        # Validation
        netG.eval()
        val_psnr, val_ssim = 0.0, 0.0
        with torch.no_grad():
            for lr, hr in val_loader:
                lr, hr = lr.to(device), hr.to(device)
                sr = netG(lr)
                p, s = compute_metrics(sr, hr)
                val_psnr += p
                val_ssim += s
        val_psnr /= len(val_loader)
        val_ssim /= len(val_loader)
        print(f"Validation → PSNR: {val_psnr:.4f} dB | SSIM: {val_ssim:.4f}")

        if val_psnr > best_psnr:
            best_psnr = val_psnr
            torch.save(netG.state_dict(), os.path.join(args.checkpoint_dir, 'generator_best.pth'))
            print("✅ Saved best ESRGAN generator.")

        # Save a sample grid
        sample_lr, sample_hr = next(iter(val_loader))
        sample_sr = netG(sample_lr.to(device))
        grid = vutils.make_grid(
            torch.cat([sample_lr, sample_sr.cpu(), sample_hr], dim=0),
            nrow=3
        )
        vutils.save_image(grid, os.path.join(args.sample_dir, f'epoch_{epoch}.png'), normalize=True)

    print("✅ Training complete.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train ESRGAN (x2 upscaling) on DIV2K")
    parser.add_argument('--train-lr',     type=str, required=True, help='Path to LR train folder')
    parser.add_argument('--train-hr',     type=str, required=True, help='Path to HR train folder')
    parser.add_argument('--val-lr',       type=str, required=True, help='Path to LR val folder')
    parser.add_argument('--val-hr',       type=str, required=True, help='Path to HR val folder')
    parser.add_argument('--epochs',       type=int, default=100)
    parser.add_argument('--batch-size',   type=int, default=16)
    parser.add_argument('--lr',           type=float, default=1e-4)
    parser.add_argument('--adv-weight',   type=float, default=5e-3)
    parser.add_argument('--hr-crop',      type=int, default=96)
    parser.add_argument('--num-workers',  type=int, default=4)
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints_esrgan')
    parser.add_argument('--sample-dir',   type=str, default='samples_esrgan')
    args = parser.parse_args()

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.sample_dir, exist_ok=True)
    train(args)
