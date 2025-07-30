import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchmetrics import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
import numpy as np
import matplotlib.pyplot as plt

import config
from dataset import MUCADDataset
from models import UNetGenerator
from utils import denorm_msi, select_msi_bands_for_rgb_display

def main():
    # --- Load Test Data ---
    full_dataset = MUCADDataset(config.DATASET_ROOT_PATH, config.IMG_HEIGHT, config.IMG_WIDTH, config.MSI_CHANNELS)
    all_items = full_dataset.all_data_items_paths
    total_samples = len(all_items)
    train_size = int(config.TRAIN_SPLIT * total_samples)
    val_size = int(config.VAL_SPLIT * total_samples)
    test_items = all_items[train_size+val_size:]

    test_dataset = MUCADDataset(config.DATASET_ROOT_PATH, config.IMG_HEIGHT, config.IMG_WIDTH, config.MSI_CHANNELS, test_items, "test")
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

    # --- Load Model ---
    generator = UNetGenerator(config.MSI_CHANNELS, config.SWIN_MODEL_NAME).to(config.DEVICE)
    generator.load_state_dict(torch.load(config.BEST_GENERATOR_MODEL_PATH))
    generator.eval()

    print("\n--- Evaluating on Test Set ---")
    psnr_metric = PeakSignalNoiseRatio().to(config.DEVICE)
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(config.DEVICE)
    mse_values, psnr_values, ssim_values, results_to_plot = [], [], [], []

    with torch.no_grad():
        for test_rgb, test_msi_real in test_loader:
            test_rgb = test_rgb.to(config.DEVICE)
            test_msi_real = test_msi_real.to(config.DEVICE)
            
            test_msi_gen = generator(test_rgb)
            
            msi_gen_denorm = denorm_msi(test_msi_gen)
            
            mse_values.append(nn.functional.mse_loss(msi_gen_denorm, test_msi_real).item())
            psnr_values.append(psnr_metric(msi_gen_denorm, test_msi_real).item())
            ssim_values.append(ssim_metric(msi_gen_denorm, test_msi_real).item())
            
            if len(results_to_plot) < 3:
                results_to_plot.append({
                    'rgb': test_rgb[0].cpu(), 'real_msi': test_msi_real[0].cpu(), 'gen_msi': msi_gen_denorm[0].cpu()
                })

    if mse_values:
        print(f"\nAverage Test Metrics:")
        print(f"  - MSE:  {np.mean(mse_values):.4f}")
        print(f"  - PSNR: {np.mean(psnr_values):.2f} dB")
        print(f"  - SSIM: {np.mean(ssim_values):.4f}")

    if results_to_plot:
        print("\nPlotting results...")
        plt.figure(figsize=(18, 6 * len(results_to_plot)))
        for i, data in enumerate(results_to_plot):
            ax = plt.subplot(len(results_to_plot), 3, i * 3 + 1)
            plt.imshow(data['rgb'].permute(1, 2, 0))
            ax.set_title("Input RGB")
            ax.axis("off")
            
            ax = plt.subplot(len(results_to_plot), 3, i * 3 + 2)
            plt.imshow(select_msi_bands_for_rgb_display(data['real_msi']))
            ax.set_title("Real MSI (False Color)")
            ax.axis("off")
            
            ax = plt.subplot(len(results_to_plot), 3, i * 3 + 3)
            plt.imshow(select_msi_bands_for_rgb_display(data['gen_msi']))
            ax.set_title("Generated MSI (False Color)")
            ax.axis("off")
        
        plt.tight_layout()
        plt.show()

if __name__ == '__main__':
    main()