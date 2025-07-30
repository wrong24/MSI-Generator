import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchmetrics import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
import random

import config
from dataset import MUCADDataset
from models import UNetGenerator, Discriminator
from utils import calculate_gradient_penalty, denorm_msi

def run_validation_epoch(generator, discriminator, val_loader, recon_loss_fn, ssim_loss_metric, device):
    generator.eval()
    discriminator.eval()
    total_val_g_loss, total_val_psnr, total_val_ssim = 0, 0, 0
    psnr_metric = PeakSignalNoiseRatio().to(device)

    with torch.no_grad():
        for rgb_val, msi_real_val in val_loader:
            rgb_val, msi_real_val = rgb_val.to(device), msi_real_val.to(device)
            msi_gen_val = generator(rgb_val)
            
            g_loss_adv_val = -discriminator(msi_gen_val).mean()
            g_loss_recon_val = recon_loss_fn(msi_gen_val, msi_real_val)
            msi_gen_denorm = denorm_msi(msi_gen_val)
            ssim_score_val = ssim_loss_metric(msi_gen_denorm, msi_real_val)
            g_loss_ssim_val = 1.0 - ssim_score_val
            g_loss_val = g_loss_adv_val + (config.LAMBDA_RECON * g_loss_recon_val) + (config.LAMBDA_SSIM * g_loss_ssim_val)
            total_val_g_loss += g_loss_val.item()
            
            total_val_psnr += psnr_metric(msi_gen_denorm, msi_real_val).item()
            total_val_ssim += ssim_score_val.item()

    avg_val_g_loss = total_val_g_loss / len(val_loader)
    avg_val_psnr = total_val_psnr / len(val_loader)
    avg_val_ssim = total_val_ssim / len(val_loader)
    generator.train()
    discriminator.train()
    return avg_val_g_loss, avg_val_psnr, avg_val_ssim

def main():
    # --- Dataloaders ---
    try:
        full_dataset = MUCADDataset(config.DATASET_ROOT_PATH, config.IMG_HEIGHT, config.IMG_WIDTH, config.MSI_CHANNELS)
        all_items = full_dataset.all_data_items_paths
        if not all_items: raise FileNotFoundError("MUCADDataset found no items.")
        total_samples = len(all_items)
        random.shuffle(all_items)
        train_size = int(config.TRAIN_SPLIT * total_samples)
        val_size = int(config.VAL_SPLIT * total_samples)
        train_items = all_items[:train_size]
        val_items = all_items[train_size:train_size+val_size]
        test_items = all_items[train_size+val_size:]

        train_dataset = MUCADDataset(config.DATASET_ROOT_PATH, config.IMG_HEIGHT, config.IMG_WIDTH, config.MSI_CHANNELS, train_items, "train")
        val_dataset = MUCADDataset(config.DATASET_ROOT_PATH, config.IMG_HEIGHT, config.IMG_WIDTH, config.MSI_CHANNELS, val_items, "validation")
        
        train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
        
        print(f"DataLoaders created: Train {len(train_dataset)}, Val {len(val_dataset)}, Test {len(test_items)}")
    except Exception as e:
        print(f"Could not create dataset due to: {e}.")
        return

    # --- Initialize Models, Losses, and Optimizers ---
    generator = UNetGenerator(config.MSI_CHANNELS, config.SWIN_MODEL_NAME).to(config.DEVICE)
    discriminator = Discriminator(config.MSI_CHANNELS, config.IMG_HEIGHT).to(config.DEVICE)

    reconstruction_loss_l1 = nn.L1Loss().to(config.DEVICE)
    ssim_loss_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(config.DEVICE)

    optimizer_G = optim.Adam(generator.parameters(), lr=config.LR_G, betas=(config.BETA1, 0.9))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=config.LR_D, betas=(config.BETA1, 0.9))

    print("Models, losses, and optimizers initialized.")

    # --- Main Training Loop ---
    print("\n--- Training with Combined L1 + SSIM Loss ---")
    GRAD_CLIP_VALUE = 1.0
    best_val_loss = float('inf')

    for epoch in range(config.NUM_EPOCHS):
        generator.train()
        discriminator.train()
        for i, (real_rgb_imgs, real_msi_imgs) in enumerate(train_loader):
            real_rgb_imgs = real_rgb_imgs.to(config.DEVICE)
            real_msi_imgs = real_msi_imgs.to(config.DEVICE)
            
            # Train Discriminator
            optimizer_D.zero_grad()
            generated_msi_d = generator(real_rgb_imgs).detach()
            d_real_score = discriminator(real_msi_imgs)
            d_fake_score = discriminator(generated_msi_d)
            gp = calculate_gradient_penalty(discriminator, real_msi_imgs.data, generated_msi_d.data, config.DEVICE)
            d_loss = d_fake_score.mean() - d_real_score.mean() + config.LAMBDA_GP * gp
            d_loss.backward()
            optimizer_D.step()

            # Train Generator
            if i % config.N_CRITIC == 0:
                optimizer_G.zero_grad()
                generated_msi_g = generator(real_rgb_imgs)
                
                g_loss_adv = -discriminator(generated_msi_g).mean()
                g_loss_recon = reconstruction_loss_l1(generated_msi_g, real_msi_imgs)
                generated_msi_denorm = denorm_msi(generated_msi_g)
                ssim_score = ssim_loss_metric(generated_msi_denorm, real_msi_imgs)
                g_loss_ssim = 1.0 - ssim_score
                
                g_loss = g_loss_adv + (config.LAMBDA_RECON * g_loss_recon) + (config.LAMBDA_SSIM * g_loss_ssim)
                
                if not torch.isnan(g_loss):
                    g_loss.backward()
                    torch.nn.utils.clip_grad_norm_(generator.parameters(), GRAD_CLIP_VALUE)
                    optimizer_G.step()
        
        # Validation step
        val_g_loss, val_psnr, val_ssim = run_validation_epoch(generator, discriminator, val_loader, reconstruction_loss_l1, ssim_loss_metric, config.DEVICE)
        
        print(f"[Epoch {epoch+1}/{config.NUM_EPOCHS}] [D loss: {d_loss.item():.3f}] [G loss: {g_loss.item():.3f}] -- "
              f"[Val Loss: {val_g_loss:.3f}] [Val PSNR: {val_psnr:.2f}] [Val SSIM: {val_ssim:.4f}]")
        
        # Save best model
        if val_g_loss < best_val_loss:
            best_val_loss = val_g_loss
            torch.save(generator.state_dict(), config.BEST_GENERATOR_MODEL_PATH)
            print(f"    -> New best model saved with validation loss: {best_val_loss:.4f}")

    print("--- Training Complete ---")

if __name__ == '__main__':
    main()