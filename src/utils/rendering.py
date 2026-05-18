import torch
from gsplat import rasterization, rasterization_2dgs
import torch.nn.functional as F
from uco3d.dataset_utils.utils import opencv_cameras_projection_from_uco3d


def get_camera_params(sample, device, ret_normals=False):
    gt_image = sample["image_rgb"].to(device)   
    fg_mask  = sample["fg_probability"].to(device)  
    h, w     = gt_image.shape[-2:]
    image_size_tensor = torch.tensor([[h, w]], dtype=torch.float32)
    Rcv, tvec, K = opencv_cameras_projection_from_uco3d(
        sample["camera"], image_size=image_size_tensor,
    )
    viewmats = torch.eye(4)[None].to(device)
    viewmats[:, :3, :4] = torch.cat([Rcv, tvec[..., None]], dim=-1)
    K = K.to(device)

    if ret_normals:
        R_c2w = viewmats[:, :3, :3].transpose(1, 2)        # (1, 3, 3)
        depth_map  = sample["depth_map"].to(device)         # (1, 1, H, W)
        depth_mask = sample["depth_mask"].to(device)        # (1, 1, H, W)
        scale_adj  = sample["depth_scale_adjustment"]
        if hasattr(scale_adj, "to"):
            scale_adj = scale_adj.to(device).view(-1, 1, 1, 1).float()
        else:
            scale_adj = float(scale_adj)
        depth_map  = depth_map * scale_adj
        gt_normals_depth = depth_to_normals(depth_map, K, depth_mask)  # (1,3,H,W)
        return viewmats, K, gt_image, fg_mask, h, w, R_c2w, gt_normals_depth

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


def render_2dgs(means, quats, scales2d, opacities, sh0, viewmats, K, h, w, device, ret_normals=False):
    out = rasterization_2dgs(
        means       = means,
        quats       = F.normalize(quats, p=2, dim=-1),
        scales      = torch.cat([
                          torch.exp(scales2d),
                          torch.full((scales2d.shape[0], 1), 1e-4, device=scales2d.device),
                      ], dim=-1),                        # (N, 3)
        opacities   = torch.sigmoid(opacities.flatten()),
        colors      = sh0.unsqueeze(1),                  # (N, 1, 3)
        viewmats    = viewmats,
        Ks          = K,
        width       = w,
        height      = h,
        sh_degree   = 0,
        near_plane  = 0.01,
        backgrounds = torch.zeros(1, 3, device=device),
        packed      = False,
        absgrad     = False,
    )
    rgb = out[0].permute(0, 3, 1, 2)                    # (1, 3, H, W) always first

    normals = None
    for tensor in out[1:]:
        t = tensor.permute(0, 3, 1, 2) if tensor.dim() == 4 else tensor
        if t.shape[1] == 3:
            normals = t
            break
    
    if normals is not None and ret_normals:
        return rgb, normals
    
    return rgb


def depth_to_normals(depth: torch.Tensor, K: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    B, _, H, W = depth.shape
    device = depth.device

    fx = K[:, 0, 0].view(B, 1, 1)   # (B,1,1)
    fy = K[:, 1, 1].view(B, 1, 1)
    cx = K[:, 0, 2].view(B, 1, 1)
    cy = K[:, 1, 2].view(B, 1, 1)

    u = torch.arange(W, dtype=torch.float32, device=device).view(1, 1, W)
    v = torch.arange(H, dtype=torch.float32, device=device).view(1, H, 1)

    Z = depth[:, 0]                  # (B, H, W)
    X = (u - cx) * Z / fx
    Y = (v - cy) * Z / fy

    xyz = torch.stack([X, Y, Z], dim=1)   # (B, 3, H, W)

    def cdiff(t, dim):
        d = torch.zeros_like(t)
        if dim == 2:
            d[:, :, 1:-1] = t[:, :, 2:] - t[:, :, :-2]
            d[:, :,  0]   = t[:, :, 1]  - t[:, :,  0]
            d[:, :, -1]   = t[:, :, -1] - t[:, :, -2]
        else:
            d[:, :, :, 1:-1] = t[:, :, :, 2:] - t[:, :, :, :-2]
            d[:, :, :,  0]   = t[:, :, :, 1]  - t[:, :, :,  0]
            d[:, :, :, -1]   = t[:, :, :, -1] - t[:, :, :, -2]
        return d

    dPdu = cdiff(xyz, dim=3)   # (B, 3, H, W)  horizontal tangent
    dPdv = cdiff(xyz, dim=2)   # (B, 3, H, W)  vertical tangent

    # cross product: (B, 3, H, W)
    normals = torch.cross(
        dPdu.permute(0, 2, 3, 1),
        dPdv.permute(0, 2, 3, 1),
        dim=-1,
    ).permute(0, 3, 1, 2)

    valid_f = (valid > 0).float()                            # (B, 1, H, W)
    normals = normals * valid_f

    norm_len = normals.norm(dim=1, keepdim=True).clamp(min=1e-6)
    normals  = (normals / norm_len) * valid_f                # unit, masked

    return normals