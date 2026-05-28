import torch
import os
from torch.optim import AdamW
import wandb
from uco3d.dataset_utils.data_types import GaussianSplats, Cameras
from radiance_field_project.data.datamodule import UCO3DDataModule
from radiance_field_project.utils.splat_visualizer import SplatRenderVisualizer
from radiance_field_project.utils.rendering import get_camera_params, render
from radiance_field_project.utils.losses import l2_loss
from radiance_field_project.models.minipointnet import MiniPointNet3DGSNetwork

os.environ["UCO3D_DATASET_ROOT"] = "/mnt/hdd/lkuech/repos/radiance-field-project/data/uco3d_data_raw"

# ── hyperparams ───────────────────────────────────────────────────────────────
EPOCHS        = 600
LR            = 1e-4
NUM_SEQUENCES = 5
NUM_VIEWS     = 10
LOG_EVERY     = 20
MAX_POINTS    = 150_000
MAX_SPLATS    = 500_000
DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LOG_DIR       = "logs_minipointnet"

print(f"Device: {DEVICE}  |  sequences: {NUM_SEQUENCES}  |  views/seq: {NUM_VIEWS}")


# ── helpers ───────────────────────────────────────────────────────────────────

def _slice_camera(cam: Cameras, i: int) -> Cameras:
    img_size = cam.image_size
    if torch.is_tensor(img_size):
        img_size = img_size[i : i + 1]
    return Cameras(
        R=cam.R[i : i + 1],
        T=cam.T[i : i + 1],
        focal_length=cam.focal_length[i : i + 1],
        principal_point=cam.principal_point[i : i + 1],
        colmap_distortion_coeffs=cam.colmap_distortion_coeffs[i : i + 1],
        device=cam.device,
        in_ndc=cam.in_ndc,
        image_size=img_size,
    )


def slice_frame(batch: dict, i: int) -> dict:
    dsa = batch["depth_scale_adjustment"]
    return {
        "image_rgb":              batch["image_rgb"][i : i + 1],
        "fg_probability":         batch["fg_probability"][i : i + 1],
        "camera":                 _slice_camera(batch["camera"], i),
        "depth_map":              batch["depth_map"][i : i + 1],
        "depth_mask":             batch["depth_mask"][i : i + 1],
        "depth_scale_adjustment": dsa[i : i + 1] if torch.is_tensor(dsa) else dsa,
    }


def subsample_gaussians(g: dict, max_points: int) -> dict:
    if g["means"].shape[0] <= max_points:
        return g
    perm = torch.randperm(g["means"].shape[0], device=g["means"].device)[:max_points]
    return {k: g[k][perm] for k in ("means", "sh0", "scales", "quats", "opacities")}


def to_splats(g: dict) -> GaussianSplats:
    return GaussianSplats(
        means=g["means"], sh0=g["sh0"], scales=g["scales"],
        quats=g["quats"], opacities=g["opacities"], shN=None, fg_mask=None,
    )


# ── data module ───────────────────────────────────────────────────────────────
dm = UCO3DDataModule(
    num_workers=4,
    image_height=512,
    image_width=512,
    load_masks=True,
    single_sequence=False,
)

# ── step 1: preload Gaussians ONCE ────────────────────────────────────────────
print("Preloading Gaussians into CPU cache...")
splat_loader = dm.splat_only_loader(max_splats=MAX_SPLATS)
splat_loader.batch_sampler._batches = [
    b[:1] for b in splat_loader.batch_sampler._batches
]

gaussian_cache = {}

for batch in splat_loader:
    seq_name = batch["sequence_name"][0] if isinstance(batch["sequence_name"], list) else batch["sequence_name"]
    raw = batch["sequence_gaussian_splats"][0]

    gaussian_cache[seq_name] = {
        "means":     raw.means.float(),
        "sh0":       raw.sh0.float(),
        "scales":    raw.scales.float(),
        "quats":     raw.quats.float(),
        "opacities": raw.opacities.float(),
    }
    print(f"  Cached Gaussians for '{seq_name}'  ({raw.means.shape[0]} points)")
    if len(gaussian_cache) >= NUM_SEQUENCES:
        break

print(f"Gaussian cache ready: {len(gaussian_cache)} sequences.\n")

# ── step 2: image-only loader ─────────────────────────────────────────────────
image_loader = dm.image_only_loader()
_img_seq_names = list(image_loader.dataset.sequence_names())
image_loader.batch_sampler._batches = [
    b[:: max(1, len(b) // NUM_VIEWS)][:NUM_VIEWS]
    for seq, b in zip(_img_seq_names, image_loader.batch_sampler._batches)
    if seq in gaussian_cache
]
print(f"Image loader ready: {len(image_loader)} sequences, up to {NUM_VIEWS} frames each.\n")

# ── model + optimiser ─────────────────────────────────────────────────────────
network   = MiniPointNet3DGSNetwork().to(DEVICE)
optimizer = AdamW(network.parameters(), lr=LR)

os.makedirs(LOG_DIR, exist_ok=True)

wandb.init(
    entity="nikolasrieger",
    project="adcv",
    name=f"minipointnet-{NUM_SEQUENCES}seq",
    config={
        "epochs": EPOCHS,
        "lr": LR,
        "num_sequences": NUM_SEQUENCES,
        "num_views": NUM_VIEWS,
        "max_points": MAX_POINTS,
        "max_splats": MAX_SPLATS,
    },
)

# ── training loop ─────────────────────────────────────────────────────────────
for epoch in range(EPOCHS):
    network.train()
    total_loss = 0.0
    n_steps    = 0

    for batch in image_loader:
        seq_name = batch["sequence_name"][0] if isinstance(batch["sequence_name"], list) else batch["sequence_name"]
        g_cpu = gaussian_cache[seq_name]

        T = batch["image_rgb"].shape[0]
        step = max(1, T // NUM_VIEWS)
        frame_indices = list(range(0, T, step))[:NUM_VIEWS]

        view_data = [
            get_camera_params(slice_frame(batch, i), DEVICE)
            for i in frame_indices
        ]

        for v_mats, v_K, v_gt_image, v_fg_mask, v_h, v_w in view_data:
            optimizer.zero_grad()

            g = {k: v.to(DEVICE) for k, v in g_cpu.items()}
            g_sub = subsample_gaussians(g, MAX_POINTS)

            recon = network.reconstruct_splats(to_splats(g_sub))
            rendered = render(
                recon.means, recon.quats, recon.scales,
                recon.opacities, recon.sh0,
                v_mats, v_K, v_h, v_w, DEVICE,
            )

            loss = l2_loss(rendered, v_gt_image, mask=v_fg_mask)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(network.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            n_steps    += 1

            del g, g_sub, recon, rendered
            torch.cuda.empty_cache()

    # ── logging ───────────────────────────────────────────────────────────────
    if epoch % LOG_EVERY == 0:
        avg_loss = total_loss / max(n_steps, 1)

        network.eval()
        for si, (seq_name, g_cpu) in enumerate(gaussian_cache.items()):
            with torch.no_grad():
                g = {k: v.to(DEVICE) for k, v in g_cpu.items()}
                g_sub = subsample_gaussians(g, MAX_POINTS)

                for batch in image_loader:
                    bn = batch["sequence_name"][0] if isinstance(batch["sequence_name"], list) else batch["sequence_name"]
                    if bn != seq_name:
                        continue

                    v0_mats, v0_K, v0_gt_image, v0_fg_mask, v0_h, v0_w = \
                        get_camera_params(slice_frame(batch, 0), DEVICE)

                    recon_eval = network.reconstruct_splats(to_splats(g_sub))
                    eval_rendered = render(
                        recon_eval.means, recon_eval.quats, recon_eval.scales,
                        recon_eval.opacities, recon_eval.sh0,
                        v0_mats, v0_K, v0_h, v0_w, DEVICE,
                    )
                    ref_rgb = render(
                        g_sub["means"], g_sub["quats"], g_sub["scales"],
                        g_sub["opacities"], g_sub["sh0"],
                        v0_mats, v0_K, v0_h, v0_w, DEVICE,
                    )
                    break

            save_path = f"{LOG_DIR}/epoch_{epoch:05d}_seq{si:02d}.png"
            SplatRenderVisualizer.save(
                save_path,
                render_colors=torch.cat([
                    v0_gt_image.permute(0, 2, 3, 1),
                    ref_rgb.permute(0, 2, 3, 1),
                    eval_rendered.permute(0, 2, 3, 1),
                ], dim=2),
                render_alphas=torch.ones(1, v0_h, v0_w * 3, 1, device=DEVICE),
            )
            wandb.log({f"images/seq{si:02d}": wandb.Image(save_path, caption="gt_rgb | ref_rgb | pred_rgb")}, step=epoch)

        wandb.log({"loss/rgb": avg_loss}, step=epoch)
        print(f"Epoch {epoch:05d}/{EPOCHS} | Avg Loss = {avg_loss:.6f} | Saved {len(gaussian_cache)} images to {LOG_DIR}/")
