import os
import argparse
from tqdm import tqdm
import torch
import torch.nn as nn
import torchvision.utils as vutils
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim

# Model imports
from model import Generator, Discriminator, VGGFeatureExtractor
from data_loader import get_dataloaders


def compute_metrics(sr, hr):
    # Assumes tensors in range [0,1], shape (B,3,H,W)
    sr_np = sr.permute(0,2,3,1).cpu().numpy()
    hr_np = hr.permute(0,2,3,1).cpu().numpy()
    psnr_vals, ssim_vals = [], []

    for i in range(sr_np.shape[0]):
        sr_img = sr_np[i]
        hr_img = hr_np[i]

        # Get dimensions
        sr_h, sr_w = sr_img.shape[:2]
        hr_h, hr_w = hr_img.shape[:2]

        # If dimensions don't match, center crop the larger image to match the smaller one
        if sr_h != hr_h or sr_w != hr_w:
            if sr_h < hr_h:
                start_h = (hr_h - sr_h) // 2
                hr_img = hr_img[start_h:start_h + sr_h, :, :]
            else:
                start_h = (sr_h - hr_h) // 2
                sr_img = sr_img[start_h:start_h + hr_h, :, :]
            
            if sr_w < hr_w:
                start_w = (hr_w - sr_w) // 2
                hr_img = hr_img[:, start_w:start_w + sr_w, :]
            else:
                start_w = (sr_w - hr_w) // 2
                sr_img = sr_img[:, start_w:start_w + hr_w, :]

        psnr_vals.append(psnr(hr_img, sr_img, data_range=1.0))
        ssim_vals.append(ssim(hr_img, sr_img, channel_axis=-1, data_range=1.0))

    return sum(psnr_vals)/len(psnr_vals), sum(ssim_vals)/len(ssim_vals)



def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Data
    train_loader, val_loader = get_dataloaders(
        args.train_lr, args.train_hr,
        args.val_lr, args.val_hr,
        hr_crop_size=args.hr_crop,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )

    # Models
    generator = Generator().to(device)
    discriminator = Discriminator().to(device)
    vgg_extractor = VGGFeatureExtractor(device=device).to(device)

    # Losses
    adversarial_criterion = nn.BCEWithLogitsLoss()
    content_criterion = nn.MSELoss()

    # Optimizers
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=args.lr, betas=(0.9, 0.999))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=args.lr, betas=(0.9, 0.999))

    # Labels
    valid_label = 1.
    fake_label = 0.

    best_psnr = 0
    for epoch in range(1, args.epochs+1):
        generator.train(); discriminator.train()
        loop = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")
        for lr, hr in loop:
            lr, hr = lr.to(device), hr.to(device)
            batch_size = lr.size(0)
            valid = torch.full((batch_size,1), valid_label, device=device)
            fake = torch.full((batch_size,1), fake_label, device=device)

            # Train Generator
            optimizer_G.zero_grad()
            sr = generator(lr)
            vgg_sr = vgg_extractor(sr)
            vgg_hr = vgg_extractor(hr)

            # Resize to match
            if vgg_sr.shape != vgg_hr.shape:
                _, _, h, w = vgg_hr.shape
                vgg_sr = nn.functional.interpolate(vgg_sr, size=(h, w), mode='bilinear', align_corners=False)

            content_loss = content_criterion(vgg_sr, vgg_hr)

            adv_loss = adversarial_criterion(discriminator(sr), valid)
            g_loss = content_loss + args.adv_weight * adv_loss
            g_loss.backward()
            optimizer_G.step()

            # Train Discriminator
            optimizer_D.zero_grad()
            real_loss = adversarial_criterion(discriminator(hr), valid)
            fake_loss = adversarial_criterion(discriminator(sr.detach()), fake)
            d_loss = (real_loss + fake_loss) / 2
            d_loss.backward()
            optimizer_D.step()

            loop.set_postfix(G_loss=g_loss.item(), D_loss=d_loss.item())

        # Validation
        generator.eval()
        val_psnr, val_ssim = 0, 0
        with torch.no_grad():
            for lr, hr in val_loader:
                lr, hr = lr.to(device), hr.to(device)
                sr = generator(lr)
                p, s = compute_metrics(sr, hr)
                val_psnr += p; val_ssim += s
            val_psnr /= len(val_loader)
            val_ssim /= len(val_loader)

        print(f"Validation PSNR: {val_psnr:.4f}, SSIM: {val_ssim:.4f}")

        # Checkpoint & samples
        if val_psnr > best_psnr:
            best_psnr = val_psnr
            torch.save(generator.state_dict(), os.path.join(args.checkpoint_dir, 'generator_best.pth'))
            print("Saved best model checkpoint.")
            
        # Save sample images
        with torch.no_grad():
            sample_lr, sample_hr = next(iter(val_loader))
            sample_lr = sample_lr.to(device)
            sample_sr = generator(sample_lr)
            
            # Resize LR to match SR size for visualization
            b, c, h, w = sample_sr.size()
            sample_lr_upscaled = nn.functional.interpolate(sample_lr, size=(h, w), mode='bicubic', align_corners=False)
            
            # Center crop HR to match SR size
            hr_h, hr_w = sample_hr.shape[2:]
            if hr_h != h or hr_w != w:
                sample_hr = nn.functional.interpolate(sample_hr, size=(h, w), mode='bicubic', align_corners=False)
                
            # Create comparison grid
            grid = vutils.make_grid(
                torch.cat([
                    sample_lr_upscaled.cpu(),
                    sample_sr.cpu(),
                    sample_hr
                ], dim=0),
                nrow=args.batch_size,
                normalize=True,
                value_range=(-1, 1)
            )
            vutils.save_image(grid, os.path.join(args.sample_dir, f'epoch_{epoch}.png'))


def parse_args():
    parser = argparse.ArgumentParser(description="Train SRGAN on DIV2K")
    parser.add_argument('--train-lr', type=str, required=True)
    parser.add_argument('--train-hr', type=str, required=True)
    parser.add_argument('--val-lr', type=str, required=True)
    parser.add_argument('--val-hr', type=str, required=True)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--adv-weight', type=float, default=1e-3)
    parser.add_argument('--hr-crop', type=int, default=96)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints')
    parser.add_argument('--sample-dir', type=str, default='samples')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.sample_dir, exist_ok=True)
    train(args)
