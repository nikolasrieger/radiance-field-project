import torch
import os 
import torch
from torch.optim import AdamW
from uco3d import render_splats
import torch.nn.functional as F
from data.gaussian_features import batch_to_features
from models.simple_autoencoder import SimpleAutoencoder
from data.datamodule import UCO3DDataModule
from data.gaussian_features import features_to_gaussians
from utils.splat_visualizer import SplatRenderVisualizer

os.environ["UCO3D_DATASET_ROOT"] = "/mnt/hdd/lkuech/repos/radiance-field-project/data/uco3d_data_raw"

EPOCHS = 20000
LR = 1E-4
BATCH_SIZE = 1

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

dm = UCO3DDataModule(
    subset_lists_filename="/mnt/hdd/lkuech/repos/radiance-field-project/data/uco3d_data_raw/set_lists/set_lists_all-categories.sqlite",
    batch_size=1,
    num_workers=0,
    image_height=512,
    image_width=512,
    load_masks=True
)

torch.manual_seed(1)
train_loader = dm.train_loader()

print(f"Dataset size: {len(train_loader.dataset)}")
print(f"Num batches:  {len(train_loader)}")

network = SimpleAutoencoder().to(device)

optimizer = AdamW(network.parameters(), lr=LR)

sample = next(iter(train_loader))
fg_mask  = sample.get("fg_probability")
fg_mask  = fg_mask.to(device) if fg_mask is not None else None 

features, batchidx = batch_to_features(sample)
features = features.to(device)

mean = features.mean(dim=0, keepdim=True)
std  = features.std(dim=0, keepdim=True).clamp(min=1e-6)
features_norm = (features - mean) / std

for epoch in range(EPOCHS):

    optimizer.zero_grad()

    recon_norm = network(features_norm)
    loss = F.mse_loss(recon_norm, features_norm)

    loss.backward()
    optimizer.step()

    if epoch % 100 == 0:
        print(f"Epoch {epoch}: Current loss = {loss.item():.6f}")


render_colors_gt, render_alphas_gt, render_info_gt = render_splats(
    cameras=sample["camera"],
    splats=sample["sequence_gaussian_splats"][0],
    render_size=[512, 512]
)

network.eval()

with torch.no_grad():
    recon_norm = network(features_norm)

sequence_gaussians_splats_reconstructed = features_to_gaussians(recon_norm * std + mean)

render_colors_recon, render_alphas_recon, render_info_recon = render_splats(
    cameras=sample["camera"],
    splats=sequence_gaussians_splats_reconstructed,
    render_size=[512, 512]
)

# Get the real photograph (Ground Truth Image) and permute to [B, H, W, 3]
real_image = sample["image_rgb"].permute(0, 2, 3, 1).to(device)
real_alpha = torch.ones_like(render_alphas_gt)

# Concatenate all THREE side-by-side: 
# 1. Real Image, 2. Rendered GT Gaussians, 3. Rendered Reconstructed Gaussians
colors_side_by_side = torch.cat([real_image, render_colors_gt, render_colors_recon], dim=2)
alphas_side_by_side = torch.cat([real_alpha, render_alphas_gt, render_alphas_recon], dim=2)

SplatRenderVisualizer.save("gt_vs_recon.png", render_colors=colors_side_by_side, render_alphas=alphas_side_by_side)