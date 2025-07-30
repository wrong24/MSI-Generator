import torch
import numpy as np

def calculate_gradient_penalty(discriminator, real_images, fake_images, device):
    B = real_images.size(0)
    alpha = torch.rand(B, 1, 1, 1, device=device).expand_as(real_images)
    interpolates = (alpha * real_images + ((1 - alpha) * fake_images)).requires_grad_(True)
    d_interpolates = discriminator(interpolates)
    gradients = torch.autograd.grad(outputs=d_interpolates, inputs=interpolates,
                                    grad_outputs=torch.ones_like(d_interpolates),
                                    create_graph=True, retain_graph=True)[0]
    gradients = gradients.view(B, -1)
    return ((gradients.norm(2, dim=1) - 1) ** 2).mean()

def denorm_msi(tensor):
    return (tensor + 1) / 2

def select_msi_bands_for_rgb_display(msi_tensor_chw):
    r, g, b = 5, 3, 1
    selected = torch.stack([msi_tensor_chw[r], msi_tensor_chw[g], msi_tensor_chw[b]], dim=-1)
    return selected.numpy().clip(0, 1)