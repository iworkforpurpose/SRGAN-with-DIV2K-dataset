import os
import argparse
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.utils as vutils
from torchvision import transforms
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim

from esrgan_model import GeneratorRRDB, Discriminator, VGGFeatureExtractor
from data_loader import get_dataloaders


def compute_metrics(sr: torch.Tensor, hr: torch.Tensor):
    """
    Compute PSNR and SSIM between super-resolved (sr) and high-resolution (hr) batches.
    Ensures tensors match in spatial dimensions by interpolation, and detaches gradients.
    """
    # Detach if gradients are tracked
    if sr.requires_grad:
        sr = sr.detach()
    if hr.requires_grad:
        hr = hr.detach()

    # Align spatial dimensions
    if sr.shape[-2:] != hr.shape[-2:]:
        hr = F.interpolate(hr, size=sr.shape[-2:], mode='bilinear', align_corners=False)

    # Convert to NumPy (NHWC)
    sr_np = sr.permute(0, 2, 3, 1).cpu().numpy()
    hr_np = hr.permute(0, 2, 3, 1).cpu().numpy()

    psnr_vals, ssim_vals = [], []
    for i in range(sr_np.shape[0]):
        psnr_vals.append(psnr(hr_np[i], sr_np[i], data_range=1.0))
        ssim_vals.append(ssim(hr_np[i], sr_np[i], data_range=1.0, channel_axis=-1))
    return sum(psnr_vals) / len(psnr_vals), sum(ssim_vals) / len(ssim_vals)


def normalize_vgg(img: torch.Tensor):
    """Normalize input image batch for VGG feature extraction"""
    mean = torch.tensor([0.485, 0.456, 0.406], device=img.device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=img.device).view(1, 3, 1, 1)
    return (img - mean) / std


def train(args):
    # Determine device and fallback if necessary
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("[WARN] CUDA requested but not available; using CPU instead.")
        device = torch.device('cpu')
    else:
        device = torch.device(args.device)
    print(f"[INFO] Using device: {device}")

    # Data loaders
    train_loader, val_loader = get_dataloaders(
        args.train_lr, args.train_hr,
        args.val_lr, args.val_hr,
        hr_crop_size=args.hr_crop,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )

    # Model setup
    netG = GeneratorRRDB().to(device)
    netD = Discriminator().to(device)
    vgg = VGGFeatureExtractor(device=device).to(device)
    for p in vgg.parameters(): p.requires_grad = False

    adv_criterion = nn.BCEWithLogitsLoss().to(device)
    content_criterion = nn.MSELoss().to(device)

    optimizerG = torch.optim.Adam(netG.parameters(), lr=args.lr, betas=(0.9, 0.999))
    optimizerD = torch.optim.Adam(netD.parameters(), lr=args.lr, betas=(0.9, 0.999))

    train_metrics = {'psnr': [], 'ssim': []}
    val_metrics   = {'psnr': [], 'ssim': []}
    best_psnr = 0.0

    for epoch in range(1, args.epochs + 1):
        netG.train(); netD.train()
        loop = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")

        epoch_train_psnr, epoch_train_ssim = 0.0, 0.0
        for lr, hr in loop:
            lr, hr = lr.to(device), hr.to(device)
            b = lr.size(0)

            # Align sizes
            if lr.shape[-2:] != hr.shape[-2:]:
                hr = F.interpolate(hr, size=lr.shape[-2:], mode='bilinear', align_corners=False)

            # Discriminator update
            optimizerD.zero_grad()
            with torch.no_grad(): sr = netG(lr)
            d_real = netD(hr)
            d_fake = netD(sr)
            d_loss = (adv_criterion(d_real - d_fake.mean(), torch.ones_like(d_real)) +
                      adv_criterion(d_fake - d_real.mean().detach(), torch.zeros_like(d_fake))) * 0.5
            d_loss.backward(); optimizerD.step()

            # Generator update
            optimizerG.zero_grad()
            sr = netG(lr)
            d_real = netD(hr).detach()
            d_fake = netD(sr)
            g_adv = adv_criterion(d_fake - d_real.mean(), torch.ones_like(d_fake))

            feat_sr = vgg(normalize_vgg(sr))
            feat_hr = vgg(normalize_vgg(hr))
            if feat_sr.shape != feat_hr.shape:
                feat_sr = F.interpolate(feat_sr, size=feat_hr.shape[-2:], mode='bilinear', align_corners=False)

            c_loss = content_criterion(feat_sr, feat_hr)
            g_loss = c_loss + args.adv_weight * g_adv
            g_loss.backward(); optimizerG.step()

            # Compute training metrics
            p, s = compute_metrics(sr, hr)
            epoch_train_psnr += p
            epoch_train_ssim += s
            loop.set_postfix(D=d_loss.item(), G=g_loss.item())

        # Log training metrics
        n_batches = len(train_loader)
        train_metrics['psnr'].append(epoch_train_psnr / n_batches)
        train_metrics['ssim'].append(epoch_train_ssim / n_batches)

        # Validation
        netG.eval()
        epoch_val_psnr, epoch_val_ssim = 0.0, 0.0
        with torch.no_grad():
            for lr, hr in val_loader:
                lr, hr = lr.to(device), hr.to(device)
                sr = netG(lr)
                p, s = compute_metrics(sr, hr)
                epoch_val_psnr += p; epoch_val_ssim += s

        val_psnr = epoch_val_psnr / len(val_loader)
        val_ssim = epoch_val_ssim / len(val_loader)
        val_metrics['psnr'].append(val_psnr)
        val_metrics['ssim'].append(val_ssim)
        print(f"Epoch {epoch} → Train PSNR: {train_metrics['psnr'][-1]:.4f}, SSIM: {train_metrics['ssim'][-1]:.4f} | "
              f"Val PSNR: {val_psnr:.4f}, SSIM: {val_ssim:.4f}")

        if val_psnr > best_psnr:
            best_psnr = val_psnr
            # Save model and optimizer together
            torch.save({
                'epoch': epoch,
                'model_state_dict': netG.state_dict(),
                'optimizer_state_dict': optimizerG.state_dict(),
                'psnr': best_psnr
            }, os.path.join(args.checkpoint_dir, 'generator_best.pth'))
            print("🚀 Saved best generator checkpoint (model + optimizer).")

        # Save sample grid
        sample_lr, sample_hr = next(iter(val_loader))
        sample_sr = netG(sample_lr.to(device))
        grid = vutils.make_grid(
            torch.cat([sample_lr, sample_sr.cpu(), sample_hr], dim=0),
            nrow=args.batch_size
        )

        # Sampling
        lr_samp, hr_samp = next(iter(val_loader))
        lr_up = F.interpolate(lr_samp, size=hr_samp.shape[-2:], mode='bilinear', align_corners=False)
        sr_up = F.interpolate(netG(lr_samp.to(device)).cpu(), size=hr_samp.shape[-2:], mode='bilinear', align_corners=False)
        grid = vutils.make_grid(torch.cat([lr_up, sr_up, hr_samp], dim=0), nrow=3)
        vutils.save_image(grid, os.path.join(args.sample_dir, f'epoch_{epoch}.png'), normalize=True)

    # Final report
    print("\nTraining complete. Metrics:")
    for i, (p, s) in enumerate(zip(train_metrics['psnr'], train_metrics['ssim']), 1):
        print(f"Epoch {i}: PSNR={p:.4f}, SSIM={s:.4f}")
    print("\nValidation Metrics:")
    for i, (p, s) in enumerate(zip(val_metrics['psnr'], val_metrics['ssim']), 1):
        print(f"Epoch {i}: PSNR={p:.4f}, SSIM={s:.4f}")

    print("✅ Done.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train ESRGAN (x2 upscaling) on DIV2K")
    parser.add_argument('--train-lr', type=str, required=True)
    parser.add_argument('--train-hr', type=str, required=True)
    parser.add_argument('--val-lr',   type=str, required=True)
    parser.add_argument('--val-hr',   type=str, required=True)
    parser.add_argument('--epochs',   type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--lr',        type=float, default=1e-4)
    parser.add_argument('--adv-weight', type=float, default=5e-3)
    parser.add_argument('--hr-crop',   type=int, default=96)
    parser.add_argument('--num-workers', type=int, default=2)
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints_esrgan')
    parser.add_argument('--sample-dir', type=str, default='samples_esrgan')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')

    args = parser.parse_args()
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.sample_dir, exist_ok=True)
    train(args)
