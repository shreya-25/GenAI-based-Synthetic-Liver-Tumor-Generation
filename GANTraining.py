"""
Streamlined Multi-GAN Pipeline for Synthetic Tumor Generation - CORRECTED VERSION
Trains DCGAN, WGAN, Aggregator, and Style Transfer networks to generate synthetic tumor patches

CRITICAL FIXES APPLIED:
1. Tumor-specific HU normalization (20-120 HU instead of -200 to 250 HU)
2. Improved tumor patch extraction with quality filtering
3. Data augmentation for better diversity
4. Better handling of tumor sizes and shapes
"""

import os, glob, logging
import numpy as np
import nibabel as nib
from scipy.ndimage import zoom, label as nd_label, center_of_mass
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
REAL_DATA_DIR = r"C:\Users\sagarwal4\Downloads\LTS_V1\Dataset\trainOriginal_65"
BASE_OUTPUT   = r"C:\Users\sagarwal4\Downloads\LTS_V1\GAN_V2_1127\SyntheticTumorsTest_FIXED"
PATCH_SIZE    = (64, 64, 64)
VOXEL_SPACING = (1.0, 1.0, 1.0)

# ⭐ CRITICAL FIX #1: Use tumor-specific HU range instead of full CT range
HU_CLIP_RANGE_TUMOR = (20, 120)  # Realistic liver tumor HU range (was -200, 250)

DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE    = 8
EPOCHS        = 100  # Increased from 50 for better convergence
LATENT_DIM    = 64
NUM_SYNTHETIC = 50

os.makedirs(BASE_OUTPUT, exist_ok=True)
CHECKPOINT_DIR = os.path.join(BASE_OUTPUT, "checkpoints")
SYNTHETIC_DIR  = os.path.join(BASE_OUTPUT, "synthetic_tumors")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(SYNTHETIC_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s: %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(BASE_OUTPUT, "training.log")),
        logging.StreamHandler()
    ]
)

# ─────────────────────────────────────────────────────────────────────────────
# DATA PREPARATION UTILITIES
# ─────────────────────────────────────────────────────────────────────────────
def load_nifti(path):
    img = nib.load(path)
    return img.get_fdata().astype(np.float32), img.affine, img.header

def save_nifti(data, affine, header, path):
    nib.save(nib.Nifti1Image(data.astype(data.dtype), affine, header), path)

def resample(data, orig_spacing, new_spacing=VOXEL_SPACING, order=1):
    factors = np.array(orig_spacing) / np.array(new_spacing)
    return zoom(data, factors, order=order)

# ⭐ CRITICAL FIX #2: Tumor-specific normalization functions
def normalize_hu_tumor(vol, clip=HU_CLIP_RANGE_TUMOR):
    """Normalize HU values specifically for tumor tissue (20-120 HU)"""
    vol = np.clip(vol, *clip)
    return 2 * ((vol - clip[0]) / (clip[1] - clip[0])) - 1

def denormalize_hu_tumor(vol_norm, clip=HU_CLIP_RANGE_TUMOR):
    """Convert normalized values back to tumor-realistic HU range"""
    return ((vol_norm + 1) / 2) * (clip[1] - clip[0]) + clip[0]

def crop_liver_roi(vol, mask):
    coords = np.argwhere(mask > 0)
    if len(coords) == 0:
        return vol, np.array([0, 0, 0])
    mins, maxs = coords.min(axis=0), coords.max(axis=0) + 1
    return vol[mins[0]:maxs[0], mins[1]:maxs[1], mins[2]:maxs[2]], mins

def extract_patch(vol, center, size=PATCH_SIZE):
    start = [int(c - s//2) for c, s in zip(center, size)]
    slices = tuple(slice(max(0, start[i]), max(0, start[i]) + size[i]) for i in range(3))
    patch = np.zeros(size, np.float32)
    region = vol[slices]
    pad_slices = tuple(slice(0, region.shape[i]) for i in range(3))
    patch[pad_slices] = region
    return patch

# ─────────────────────────────────────────────────────────────────────────────
# ⭐ CRITICAL FIX #3: IMPROVED PATCH EXTRACTION WITH QUALITY FILTERING
# ─────────────────────────────────────────────────────────────────────────────
def sample_tumor_patches_improved():
    """Extract high-quality tumor patches with filtering"""
    vols = sorted(glob.glob(os.path.join(REAL_DATA_DIR, "volume-*.nii")))
    segs = sorted(glob.glob(os.path.join(REAL_DATA_DIR, "segmentation-*.nii")))
    tumor_patches, tumor_masks = [], []
    
    # Quality filters
    MIN_TUMOR_VOXELS = 500
    MAX_TUMOR_VOXELS = 100000
    MIN_PATCH_TUMOR_CONTENT = 300
    
    logging.info(f"Extracting quality-filtered tumor patches from {len(vols)} volumes...")
    logging.info(f"Using tumor-specific HU range: {HU_CLIP_RANGE_TUMOR}")
    
    for vpath, spath in zip(vols, segs):
        logging.info(f"Processing {os.path.basename(vpath)}...")
        vol, _, hdr = load_nifti(vpath)
        seg, _, _ = load_nifti(spath)
        orig_sp = hdr.get_zooms()[:3]
        
        vol_resampled = resample(vol, orig_sp)
        vol_norm = normalize_hu_tumor(vol_resampled)
        seg_resampled = resample(seg, orig_sp, order=0)
        
        vol_roi, offset = crop_liver_roi(vol_norm, seg_resampled)
        seg_roi, _ = crop_liver_roi(seg_resampled, seg_resampled)
        
        if vol_roi.size == 0 or seg_roi.size == 0:
            continue
        
        # Label each individual tumor
        tumor_binary = (seg_roi == 2).astype(np.int32)
        labeled_tumors, num_tumors = nd_label(tumor_binary)
        
        if num_tumors == 0:
            continue
        
        logging.info(f"  Found {num_tumors} individual tumors")
        
        for tumor_id in range(1, num_tumors + 1):
            tumor_mask_3d = (labeled_tumors == tumor_id)
            tumor_size = tumor_mask_3d.sum()
            
            if tumor_size < MIN_TUMOR_VOXELS or tumor_size > MAX_TUMOR_VOXELS:
                continue
            
            center = center_of_mass(tumor_mask_3d)
            center = np.array(center).astype(int)
            
            patch = extract_patch(vol_roi, center)
            mask = extract_patch(seg_roi, center)
            
            tumor_content = (mask == 2).sum()
            if tumor_content < MIN_PATCH_TUMOR_CONTENT:
                continue
            
            tumor_patches.append(patch)
            tumor_masks.append((mask == 2).astype(np.float32))
    
    logging.info(f"✅ Extracted {len(tumor_patches)} high-quality tumor patches")
    return tumor_patches, tumor_masks

# ⭐ CRITICAL FIX #4: DATA AUGMENTATION
class TumorPatchDatasetAugmented(Dataset):
    def __init__(self, patches, masks):
        self.patches = patches
        self.masks = masks
    
    def __len__(self):
        return len(self.patches) * 3
    
    def __getitem__(self, idx):
        base_idx = idx % len(self.patches)
        patch = self.patches[base_idx].copy()
        mask = self.masks[base_idx].copy()
        
        if idx >= len(self.patches):
            # Random flips
            if np.random.rand() > 0.5:
                patch = np.flip(patch, axis=0).copy()
                mask = np.flip(mask, axis=0).copy()
            if np.random.rand() > 0.5:
                patch = np.flip(patch, axis=1).copy()
                mask = np.flip(mask, axis=1).copy()
            
            # Random rotation
            k = np.random.randint(0, 4)
            if k > 0:
                patch = np.rot90(patch, k, axes=(0, 1)).copy()
                mask = np.rot90(mask, k, axes=(0, 1)).copy()
            
            # Intensity jitter
            jitter = np.random.randn(*patch.shape) * 0.03
            patch = patch + jitter
            patch = np.clip(patch, -1, 1)
        
        return torch.from_numpy(patch)[None].float(), torch.from_numpy(mask)[None].float()

# ─────────────────────────────────────────────────────────────────────────────
# GAN ARCHITECTURES
# ─────────────────────────────────────────────────────────────────────────────
class DCGAN3DGenerator(nn.Module):
    def __init__(self):
        super().__init__()
        b = 64
        self.net = nn.Sequential(
            nn.ConvTranspose3d(LATENT_DIM, b*8, 4, 1, 0),
            nn.BatchNorm3d(b*8), nn.ReLU(True),
            nn.ConvTranspose3d(b*8, b*4, 4, 2, 1),
            nn.BatchNorm3d(b*4), nn.ReLU(True),
            nn.ConvTranspose3d(b*4, b*2, 4, 2, 1),
            nn.BatchNorm3d(b*2), nn.ReLU(True),
            nn.ConvTranspose3d(b*2, b, 4, 2, 1),
            nn.BatchNorm3d(b), nn.ReLU(True),
            nn.ConvTranspose3d(b, 1, 4, 2, 1),
            nn.Tanh()
        )
    
    def forward(self, z):
        return self.net(z)

class DCGAN3DDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        b = 64
        self.net = nn.Sequential(
            nn.Conv3d(1, b, 4, 2, 1), nn.LeakyReLU(0.2, True),
            nn.Conv3d(b, b*2, 4, 2, 1), nn.BatchNorm3d(b*2), nn.LeakyReLU(0.2, True),
            nn.Conv3d(b*2, b*4, 4, 2, 1), nn.BatchNorm3d(b*4), nn.LeakyReLU(0.2, True),
            nn.Conv3d(b*4, 1, 4, 1, 0), nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.net(x).view(-1)

class WGAN3DGenerator(DCGAN3DGenerator):
    pass

class WGAN3DCritic(nn.Module):
    def __init__(self):
        super().__init__()
        b = 64
        self.net = nn.Sequential(
            nn.Conv3d(1, b, 4, 2, 1), nn.LeakyReLU(0.2, True),
            nn.Conv3d(b, b*2, 4, 2, 1), nn.BatchNorm3d(b*2), nn.LeakyReLU(0.2, True),
            nn.Conv3d(b*2, b*4, 4, 2, 1), nn.BatchNorm3d(b*4), nn.LeakyReLU(0.2, True),
            nn.Conv3d(b*4, 1, 4, 1, 0)
        )
    
    def forward(self, x):
        return self.net(x).view(-1)

class Aggregator3D(nn.Module):
    def __init__(self):
        super().__init__()
        b = 32
        self.net = nn.Sequential(
            nn.Conv3d(3, b, 3, 1, 1), nn.LeakyReLU(0.2, True),
            nn.Conv3d(b, b, 3, 1, 1), nn.LeakyReLU(0.2, True),
            nn.Conv3d(b, 1, 3, 1, 1), nn.Tanh()
        )
    
    def forward(self, a, b, c):
        return self.net(torch.cat([a, b, c], 1))

class StyleTransfer3D(nn.Module):
    def __init__(self):
        super().__init__()
        b = 32
        self.net = nn.Sequential(
            nn.Conv3d(1, b, 3, 1, 1), nn.LeakyReLU(0.2, True),
            nn.Conv3d(b, b, 3, 1, 1), nn.LeakyReLU(0.2, True),
            nn.Conv3d(b, 1, 3, 1, 1), nn.Tanh()
        )
    
    def forward(self, x):
        return self.net(x)

class Aggregator3DDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        b = 64
        self.net = nn.Sequential(
            nn.Conv3d(1, b, 4, 2, 1), nn.LeakyReLU(0.2, True),
            nn.Conv3d(b, b*2, 4, 2, 1), nn.BatchNorm3d(b*2), nn.LeakyReLU(0.2, True),
            nn.Conv3d(b*2, b*4, 4, 2, 1), nn.BatchNorm3d(b*4), nn.LeakyReLU(0.2, True),
            nn.Conv3d(b*4, 1, 4, 1, 0), nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.net(x).view(-1)

def gradient_penalty(critic, real, fake, λ=10):
    if fake.shape != real.shape:
        fake = torch.nn.functional.interpolate(
            fake, size=real.shape[2:],
            mode='trilinear', align_corners=False
        )
    
    α = torch.rand(real.size(0), 1, 1, 1, 1, device=real.device)
    inter = (α * real + (1 - α) * fake).requires_grad_(True)
    out = critic(inter)
    grads = torch.autograd.grad(
        outputs=out, inputs=inter,
        grad_outputs=torch.ones_like(out),
        create_graph=True, retain_graph=True
    )[0]
    return λ * ((grads.view(grads.size(0), -1).norm(2, 1) - 1) ** 2).mean()

# ─────────────────────────────────────────────────────────────────────────────
# TRAINING PIPELINE
# ─────────────────────────────────────────────────────────────────────────────
def train_multi_gan(dataloader):
    logging.info("Initializing multi-GAN models...")
    
    dc1_g = DCGAN3DGenerator().to(DEVICE)
    dc1_d = DCGAN3DDiscriminator().to(DEVICE)
    dc2_g = DCGAN3DGenerator().to(DEVICE)
    dc2_d = DCGAN3DDiscriminator().to(DEVICE)
    w_g = WGAN3DGenerator().to(DEVICE)
    w_c = WGAN3DCritic().to(DEVICE)
    aggr = Aggregator3D().to(DEVICE)
    style = StyleTransfer3D().to(DEVICE)
    ag_d = Aggregator3DDiscriminator().to(DEVICE)
    
    opt = lambda p, lr: optim.Adam(p, lr=lr, betas=(0.5, 0.999))
    dc1_oG = opt(dc1_g.parameters(), 2e-4)
    dc1_oD = opt(dc1_d.parameters(), 2e-4)
    dc2_oG = opt(dc2_g.parameters(), 2e-4)
    dc2_oD = opt(dc2_d.parameters(), 2e-4)
    w_oG = opt(w_g.parameters(), 5e-5)
    w_oC = opt(w_c.parameters(), 5e-5)
    ag_oG = opt(list(aggr.parameters()) + list(style.parameters()), 2e-4)
    ag_oD = opt(ag_d.parameters(), 2e-4)
    
    bce = nn.BCELoss()
    history = {'dc1_d': [], 'dc1_g': [], 'dc2_d': [], 'dc2_g': [], 'w_c': [], 'w_g': [], 'agg_d': [], 'agg_g': []}
    
    logging.info(f"Starting training for {EPOCHS} epochs...")
    for epoch in range(EPOCHS):
        epoch_losses = {k: [] for k in history.keys()}
        
        for batch_idx, (real, _) in enumerate(dataloader):
            real = real.to(DEVICE)
            bs = real.size(0)
            
            # Train DCGAN 1
            dc1_oD.zero_grad()
            z1 = torch.randn(bs, LATENT_DIM, 1, 1, 1, device=DEVICE)
            fake1 = dc1_g(z1)
            d_real = dc1_d(real)
            d_fake = dc1_d(fake1.detach())
            d_loss = bce(d_real, torch.ones_like(d_real)) + bce(d_fake, torch.zeros_like(d_fake))
            d_loss.backward()
            dc1_oD.step()
            
            dc1_oG.zero_grad()
            d_fake = dc1_d(fake1)
            g_loss = bce(d_fake, torch.ones_like(d_fake))
            g_loss.backward()
            dc1_oG.step()
            epoch_losses['dc1_d'].append(d_loss.item())
            epoch_losses['dc1_g'].append(g_loss.item())
            
            # Train DCGAN 2
            dc2_oD.zero_grad()
            z2 = torch.randn(bs, LATENT_DIM, 1, 1, 1, device=DEVICE)
            fake2 = dc2_g(z2)
            d_real2 = dc2_d(real)
            d_fake2 = dc2_d(fake2.detach())
            d_loss2 = bce(d_real2, torch.ones_like(d_real2)) + bce(d_fake2, torch.zeros_like(d_fake2))
            d_loss2.backward()
            dc2_oD.step()
            
            dc2_oG.zero_grad()
            d_fake2 = dc2_d(fake2)
            g_loss2 = bce(d_fake2, torch.ones_like(d_fake2))
            g_loss2.backward()
            dc2_oG.step()
            epoch_losses['dc2_d'].append(d_loss2.item())
            epoch_losses['dc2_g'].append(g_loss2.item())
            
            # Train WGAN
            for _ in range(3):
                w_oC.zero_grad()
                z3 = torch.randn(bs, LATENT_DIM, 1, 1, 1, device=DEVICE)
                fake3 = w_g(z3)
                c_real = w_c(real)
                c_fake = w_c(fake3.detach())
                gp = gradient_penalty(w_c, real, fake3)
                c_loss = -(c_real.mean() - c_fake.mean()) + gp
                c_loss.backward()
                w_oC.step()
            
            w_oG.zero_grad()
            z3 = torch.randn(bs, LATENT_DIM, 1, 1, 1, device=DEVICE)
            fake3 = w_g(z3)
            g_loss3 = -w_c(fake3).mean()
            g_loss3.backward()
            w_oG.step()
            epoch_losses['w_c'].append(c_loss.item())
            epoch_losses['w_g'].append(g_loss3.item())
            
            # Train Aggregator
            ag_oD.zero_grad()
            with torch.no_grad():
                z1 = torch.randn(bs, LATENT_DIM, 1, 1, 1, device=DEVICE)
                z2 = torch.randn(bs, LATENT_DIM, 1, 1, 1, device=DEVICE)
                z3 = torch.randn(bs, LATENT_DIM, 1, 1, 1, device=DEVICE)
                agg_fake = aggr(dc1_g(z1), dc2_g(z2), w_g(z3))
                styled = style(agg_fake)
            
            ag_d_real = ag_d(real)
            ag_d_fake = ag_d(styled.detach())
            ag_d_loss = bce(ag_d_real, torch.ones_like(ag_d_real)) + bce(ag_d_fake, torch.zeros_like(ag_d_fake))
            ag_d_loss.backward()
            ag_oD.step()
            
            ag_oG.zero_grad()
            z1 = torch.randn(bs, LATENT_DIM, 1, 1, 1, device=DEVICE)
            z2 = torch.randn(bs, LATENT_DIM, 1, 1, 1, device=DEVICE)
            z3 = torch.randn(bs, LATENT_DIM, 1, 1, 1, device=DEVICE)
            agg_fake = aggr(dc1_g(z1), dc2_g(z2), w_g(z3))
            styled = style(agg_fake)
            ag_g_out = ag_d(styled)
            ag_g_loss = bce(ag_g_out, torch.ones_like(ag_g_out))
            ag_g_loss.backward()
            ag_oG.step()
            epoch_losses['agg_d'].append(ag_d_loss.item())
            epoch_losses['agg_g'].append(ag_g_loss.item())
            
            if batch_idx % 10 == 0:
                logging.info(
                    f"Epoch [{epoch+1}/{EPOCHS}] Batch [{batch_idx}/{len(dataloader)}] | "
                    f"DC1_D: {d_loss:.4f}, DC1_G: {g_loss:.4f} | "
                    f"DC2_D: {d_loss2:.4f}, DC2_G: {g_loss2:.4f} | "
                    f"W_C: {c_loss:.4f}, W_G: {g_loss3:.4f} | "
                    f"Agg_D: {ag_d_loss:.4f}, Agg_G: {ag_g_loss:.4f}"
                )
        
        for key in history.keys():
            history[key].append(np.mean(epoch_losses[key]))
        
        if (epoch + 1) % 10 == 0:
            with torch.no_grad():
                dc1_g.eval(); dc2_g.eval(); w_g.eval(); aggr.eval(); style.eval()
                z_sample = torch.randn(1, LATENT_DIM, 1, 1, 1, device=DEVICE)
                styled_sample = style(aggr(dc1_g(z_sample), dc2_g(z_sample), w_g(z_sample)))
                styled_hu = denormalize_hu_tumor(styled_sample.cpu().numpy())
                logging.info(f"Epoch {epoch+1} - Sample HU range: [{styled_hu.min():.1f}, {styled_hu.max():.1f}]")
                dc1_g.train(); dc2_g.train(); w_g.train(); aggr.train(); style.train()
    
    logging.info("Saving final model checkpoints...")
    models = {"dc1_g": dc1_g, "dc1_d": dc1_d, "dc2_g": dc2_g, "dc2_d": dc2_d,
              "w_g": w_g, "w_c": w_c, "aggr": aggr, "style": style, "ag_d": ag_d}
    for name, model in models.items():
        torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, f"{name}_final.pth"))
    
    return dc1_g, dc2_g, w_g, aggr, style, history

# ⭐ CRITICAL FIX #5: GENERATE WITH CORRECT HU RANGE
def generate_synthetic_tumors_fixed(models, num_samples=50):
    dc1_g, dc2_g, w_g, aggr, style = models
    for m in models:
        m.eval()
    
    vols = sorted(glob.glob(os.path.join(REAL_DATA_DIR, "volume-*.nii")))
    sample_img = nib.load(vols[0])
    aff, hdr = sample_img.affine, sample_img.header
    
    logging.info(f"Generating {num_samples} synthetic tumor samples...")
    hu_values_all = []
    
    for i in range(num_samples):
        with torch.no_grad():
            z = lambda: torch.randn(1, LATENT_DIM, 1, 1, 1, device=DEVICE)
            aggregated = aggr(dc1_g(z()), dc2_g(z()), w_g(z()))
            styled = style(aggregated)
            
            tumor_volume = styled.squeeze().cpu().numpy()
            tumor_mask = (tumor_volume > 0).astype(np.int16)
            tumor_hu = denormalize_hu_tumor(tumor_volume, HU_CLIP_RANGE_TUMOR)
            
            tumor_voxels = tumor_hu[tumor_mask > 0]
            if len(tumor_voxels) > 0:
                hu_values_all.extend(tumor_voxels.tolist())
        
        save_nifti(tumor_hu, aff, hdr, os.path.join(SYNTHETIC_DIR, f"synthetic_tumor_{i:03d}.nii"))
        save_nifti(tumor_mask, aff, hdr, os.path.join(SYNTHETIC_DIR, f"synthetic_mask_{i:03d}.nii"))
        
        if (i + 1) % 10 == 0:
            logging.info(f"Generated {i+1}/{num_samples} samples")
    
    if len(hu_values_all) > 0:
        logging.info(f"\n{'='*70}")
        logging.info(f"GENERATED TUMOR HU STATISTICS:")
        logging.info(f"  Mean: {np.mean(hu_values_all):.1f} HU")
        logging.info(f"  Std:  {np.std(hu_values_all):.1f} HU")
        logging.info(f"  Range: [{np.min(hu_values_all):.1f}, {np.max(hu_values_all):.1f}] HU")
        logging.info(f"{'='*70}\n")
    
    logging.info(f"✅ Saved {num_samples} synthetic tumors to {SYNTHETIC_DIR}")

def plot_training_results(history):
    epochs = range(1, len(history['dc1_d']) + 1)
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Multi-GAN Training Results (FIXED)', fontsize=16)
    
    axes[0, 0].plot(epochs, history['dc1_d'], 'b-', label='Discriminator', linewidth=2)
    axes[0, 0].plot(epochs, history['dc1_g'], 'r-', label='Generator', linewidth=2)
    axes[0, 0].set_title('DCGAN 1'); axes[0, 0].legend(); axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].plot(epochs, history['dc2_d'], 'b-', label='Discriminator', linewidth=2)
    axes[0, 1].plot(epochs, history['dc2_g'], 'r-', label='Generator', linewidth=2)
    axes[0, 1].set_title('DCGAN 2'); axes[0, 1].legend(); axes[0, 1].grid(True, alpha=0.3)
    
    axes[0, 2].plot(epochs, history['w_c'], 'b-', label='Critic', linewidth=2)
    axes[0, 2].plot(epochs, history['w_g'], 'r-', label='Generator', linewidth=2)
    axes[0, 2].set_title('WGAN'); axes[0, 2].legend(); axes[0, 2].grid(True, alpha=0.3)
    
    axes[1, 0].plot(epochs, history['agg_d'], 'b-', label='Discriminator', linewidth=2)
    axes[1, 0].plot(epochs, history['agg_g'], 'r-', label='Generator', linewidth=2)
    axes[1, 0].set_title('Aggregator'); axes[1, 0].legend(); axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].plot(epochs, history['dc1_d'], label='DCGAN1 D')
    axes[1, 1].plot(epochs, history['dc2_d'], label='DCGAN2 D')
    axes[1, 1].plot(epochs, history['agg_d'], label='Agg D')
    axes[1, 1].set_title('Discriminators'); axes[1, 1].legend(); axes[1, 1].grid(True, alpha=0.3)
    
    axes[1, 2].plot(epochs, history['dc1_g'], label='DCGAN1 G')
    axes[1, 2].plot(epochs, history['dc2_g'], label='DCGAN2 G')
    axes[1, 2].plot(epochs, history['agg_g'], label='Agg G')
    axes[1, 2].set_title('Generators'); axes[1, 2].legend(); axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(BASE_OUTPUT, "training_loss_curves.png"), dpi=150)
    plt.close()

# ─────────────────────────────────────────────────────────────────────────────
# MAIN EXECUTION
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.info("\n" + "="*80)
    logging.info("FIXED SYNTHETIC TUMOR GENERATION PIPELINE")
    logging.info(f"✅ Using tumor-specific HU range: {HU_CLIP_RANGE_TUMOR}")
    logging.info(f"✅ Training for {EPOCHS} epochs")
    logging.info("="*80 + "\n")
    
    tumor_patches, tumor_masks = sample_tumor_patches_improved()
    
    if len(tumor_patches) == 0:
        logging.error("No tumor patches extracted! Check your data.")
        exit(1)
    
    dataset = TumorPatchDatasetAugmented(tumor_patches, tumor_masks)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    logging.info(f"Dataset: {len(tumor_patches)} base patches → {len(dataset)} with augmentation\n")
    
    *trained_models, history = train_multi_gan(dataloader)
    plot_training_results(history)
    generate_synthetic_tumors_fixed(trained_models, num_samples=NUM_SYNTHETIC)
    
    logging.info("\n" + "="*80)
    logging.info("✅ PIPELINE COMPLETE!")
    logging.info(f"Output: {BASE_OUTPUT}")
    logging.info("\nNEXT: Run comparison analysis to verify HU distribution (20-120 HU)")
    logging.info("="*80 + "\n")