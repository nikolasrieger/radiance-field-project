import torch
import torch.nn as nn
import math
from diff_gaussian_rasterization import GaussianRasterizer, GaussianRasterizationSettings


def _gaussian_window(size, sigma, channels, device):
    coords = torch.arange(size, dtype=torch.float32, device=device) - size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g /= g.sum()
    window_1d = g.unsqueeze(1) 
    window_2d = window_1d @ window_1d.t() 
    window = window_2d.unsqueeze(0).unsqueeze(0).expand(channels, 1, size, size)
    return window.contiguous()


def ssim(img1, img2, window_size = 11, C1 = 0.02, C2 = 0.06):
    if img1.dim() == 3:
        img1 = img1.unsqueeze(0)
        img2 = img2.unsqueeze(0)

    channels = img1.shape[1]
    window = _gaussian_window(window_size, 1.5, channels, img1.device)
    pad = window_size // 2

    mu1 = nn.functional.conv2d(img1, window, padding=pad, groups=channels)
    mu2 = nn.functional.conv2d(img2, window, padding=pad, groups=channels)

    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = nn.functional.conv2d(img1 * img1, window, padding=pad, groups=channels) - mu1_sq
    sigma2_sq = nn.functional.conv2d(img2 * img2, window, padding=pad, groups=channels) - mu2_sq
    sigma12 = nn.functional.conv2d(img1 * img2, window, padding=pad, groups=channels) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


class DifferentiableRenderer(nn.Module):
    def __init__(self, bg_color=(0.0, 0.0, 0.0), znear=0.01, zfar=100.0):
        super().__init__()
        self.register_buffer("bg_color", torch.tensor(bg_color, dtype=torch.float32))
        self.znear = znear
        self.zfar = zfar

    @staticmethod
    def _ndc_focal_to_pixels(focal_ndc, image_size):
        s = min(image_size[0], image_size[1])
        return focal_ndc * s / 2.0

    @staticmethod
    def _ndc_pp_to_pixels(pp_ndc, image_size):
        s = min(image_size[0], image_size[1])
        cx = -pp_ndc[0] * s / 2.0 + image_size[1] / 2.0
        cy = -pp_ndc[1] * s / 2.0 + image_size[0] / 2.0
        return cx, cy

    @staticmethod
    def create_view_matrix(R, T):
        view = torch.eye(4, dtype=R.dtype, device=R.device)
        view[:3, :3] = R.T
        view[:3, 3] = T
        return view.T.contiguous()

    @staticmethod
    def get_projection_matrix(znear, zfar, fov_x, fov_y, device=None):
        tan_half_x = math.tan(fov_x * 0.5)
        tan_half_y = math.tan(fov_y * 0.5)

        P = torch.zeros(4, 4, dtype=torch.float32, device=device)
        P[0, 0] = 1.0 / tan_half_x
        P[1, 1] = 1.0 / tan_half_y
        P[2, 2] = zfar / (zfar - znear)
        P[2, 3] = -(zfar * znear) / (zfar - znear)
        P[3, 2] = 1.0
        return P

    @staticmethod
    def get_camera_center(R, T):
        return -T @ R

    def _render_single(self, means3D, opacities, scales, rotations, shs, camera):
        device = means3D.device
        H = int(camera["height"])
        W = int(camera["width"])

        R = camera["R"].to(device)
        T = camera["T"].to(device) 

        fx = self._ndc_focal_to_pixels(camera["fx"], (H, W))
        fy = self._ndc_focal_to_pixels(camera["fy"], (H, W))

        fov_x = 2.0 * math.atan(W / (2.0 * fx))
        fov_y = 2.0 * math.atan(H / (2.0 * fy))

        view_matrix = self.create_view_matrix(R, T)
        proj_matrix = self.get_projection_matrix(self.znear, self.zfar, fov_x, fov_y, device=device)
        full_proj = view_matrix @ proj_matrix
        campos = self.get_camera_center(R, T)

        settings = GaussianRasterizationSettings(
            image_height=H,
            image_width=W,
            tanfovx=math.tan(fov_x * 0.5),
            tanfovy=math.tan(fov_y * 0.5),
            bg=self.bg_color.to(device),
            scale_modifier=1.0,
            viewmatrix=view_matrix,
            projmatrix=full_proj,
            sh_degree=3,
            campos=campos,
            prefiltered=False,
            debug=False,
        )
        rasterizer = GaussianRasterizer(raster_settings=settings)

        means2D = torch.zeros(means3D.shape[0], 2, dtype=torch.float32, device=device, requires_grad=True,)

        rendered_image, _ = rasterizer(
            means3D=means3D,
            means2D=means2D,
            shs=shs,
            colors_precomp=None,
            opacities=opacities,
            scales=scales,
            rotations=rotations,
            cov3D_precomp=None,
        )
        return rendered_image  

    def forward(self, means3D, opacities, scales, rotations, shs, camera):
        if not isinstance(means3D, (list, tuple)):
            return self._render_single(means3D, opacities, scales, rotations, shs, camera)

        batch_size = len(means3D)
        renders = []
        for b in range(batch_size):
            cam_b = {
                k: (v[b] if isinstance(v, torch.Tensor) else v)
                for k, v in camera.items()
            }
            renders.append(
                self._render_single(
                    means3D[b], opacities[b], scales[b], rotations[b], shs[b], cam_b
                )
            )
        return renders


def train_loss(rendered_img, gt_img, mask, lambda_ssim=0.2):
    masked_render = rendered_img * mask
    masked_gt = gt_img * mask

    l1 = nn.functional.l1_loss(masked_render, masked_gt)
    ssim_val = ssim(masked_render, masked_gt)
    dssim = 1.0 - ssim_val

    return (1.0 - lambda_ssim) * l1 + lambda_ssim * dssim
