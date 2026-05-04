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
        # UCO3D NDC focal length to pixel focal length
        s = min(image_size[0], image_size[1])
        return focal_ndc * s / 2.0

    @staticmethod
    def create_view_matrix(R, T):
        # Build a 4x4 World-to-View matrix
        view = torch.eye(4, dtype=R.dtype, device=R.device)
        view[:3, :3] = R.t() # Transpose R for World-to-Camera
        view[:3, 3]  = T
        return view.contiguous()

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
        # Camera position in world space: C = -R^T * T
        return -R.t() @ T

    def _render_single(self, means3D, opacities, scales, rotations, shs, camera):
        device = means3D.device
        H = int(camera["height"])
        W = int(camera["width"])

        # 1. Get raw UCO3D parameters
        R = camera["R"].to(device)
        T = camera["T"].to(device)

        # 2. COORDINATE TRANSFORM: OpenCV (UCO3D) -> OpenGL (Rasterizer)
        # Flip Y and Z axes of the camera system
        R_gl = R.clone()
        T_gl = T.clone()
        R_gl[1:3, :] *= -1 
        T_gl[1:3] *= -1

        # 3. Focal length conversion (NDC to Pixels)
        fx = self._ndc_focal_to_pixels(camera["fx"].to(device), (H, W))
        fy = self._ndc_focal_to_pixels(camera["fy"].to(device), (H, W))

        # 4. Field of View calculation
        fov_x = 2.0 * math.atan(W / (2.0 * fx))
        fov_y = 2.0 * math.atan(H / (2.0 * fy))

        # 5. Build Matrices
        view_matrix = self.create_view_matrix(R_gl, T_gl)
        proj_matrix = self.get_projection_matrix(self.znear, self.zfar, fov_x, fov_y, device=device)
        full_proj = view_matrix @ proj_matrix
        campos = self.get_camera_center(R_gl, T_gl)

        settings = GaussianRasterizationSettings(
            image_height=H,
            image_width=W,
            tanfovx=math.tan(fov_x * 0.5),
            tanfovy=math.tan(fov_y * 0.5),
            bg=self.bg_color.to(device),
            scale_modifier=1.0,
            viewmatrix=view_matrix.t(), # Rasterizer expects transposed viewmatrix
            projmatrix=full_proj.t(),   # Rasterizer expects transposed projmatrix
            sh_degree=0,
            campos=campos,
            prefiltered=False,
            debug=False,
        )
        
        rasterizer = GaussianRasterizer(raster_settings=settings)

        # Ensure means2D is correctly initialized for the rasterizer
        means2D = torch.zeros_like(means3D, dtype=torch.float32, device=device, requires_grad=True)

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
        if not isinstance(means3D, (list, tuple)) and means3D.dim() == 2:
            return self._render_single(means3D, opacities, scales, rotations, shs, camera)

        # Batch rendering logic
        batch_size = len(means3D)
        renders = []
        for b in range(batch_size):
            cam_b = {
                k: (v[b] if isinstance(v, torch.Tensor) and v.dim() > 0 else v)
                for k, v in camera.items()
            }
            renders.append(
                self._render_single(
                    means3D[b], opacities[b], scales[b], rotations[b], shs[b], cam_b
                )
            )
        return torch.stack(renders) if isinstance(renders[0], torch.Tensor) else renders

def train_loss(rendered_img, gt_img, mask, lambda_ssim=0.2):
    # Ensure all have same batch dimension
    if rendered_img.dim() == 3: rendered_img = rendered_img.unsqueeze(0)
    if gt_img.dim() == 3: gt_img = gt_img.unsqueeze(0)
    if mask.dim() == 3: mask = mask.unsqueeze(0)

    masked_render = rendered_img * mask
    masked_gt = gt_img * mask

    l1 = nn.functional.l1_loss(masked_render, masked_gt)
    ssim_val = ssim(masked_render, masked_gt)
    dssim = 1.0 - ssim_val

    return (1.0 - lambda_ssim) * l1 + lambda_ssim * dssim