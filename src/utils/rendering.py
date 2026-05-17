import torch
from gsplat import rasterization
import torch.nn.functional as F
from uco3d.dataset_utils.utils import opencv_cameras_projection_from_uco3d


def get_camera_params(sample, device):
    gt_image = sample["image_rgb"].to(device)          # (1, 3, H, W)
    fg_mask = sample["fg_probability"].to(device)      # (1, 1, H, W)

    h, w = gt_image.shape[-2:]
    image_size_tensor = torch.tensor([[h, w]], dtype=torch.float32)
    Rcv, tvec, K = opencv_cameras_projection_from_uco3d(
        sample["camera"], image_size=image_size_tensor,
    )
    viewmats = torch.eye(4)[None].to(device)
    viewmats[:, :3, :4] = torch.cat([Rcv, tvec[..., None]], dim=-1)
    K = K.to(device)                                   # (1, 3, 3)

    return viewmats, K, gt_image, fg_mask, h, w


def render(means, quats, scales, opacities, sh0, viewmats, K, h, w, device):
    imgs, _, _ = rasterization(
        means          = means,
        quats          = F.normalize(quats, p=2, dim=-1),
        scales         = torch.exp(scales),
        opacities      = torch.sigmoid(opacities.flatten()),
        colors         = sh0.unsqueeze(1), 
        viewmats       = viewmats,
        Ks             = K,
        width          = w,
        height         = h,
        sh_degree      = 0,
        near_plane     = 0.01,
        backgrounds    = torch.zeros(1, 3, device=device),
        packed         = False,
        absgrad        = False,
        sparse_grad    = False,
        rasterize_mode = "classic",
        distributed    = False,
    )
    return imgs.permute(0, 3, 1, 2)