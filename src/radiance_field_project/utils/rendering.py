import torch
import torch.nn.functional as F
from gsplat import rasterization, rasterization_2dgs
from uco3d import GaussianSplats
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


def _make_linprim_camera(
    viewmats: torch.Tensor,
    Ks: torch.Tensor,
    width: int,
    height: int,
    znear: float = 0.01,
    zfar: float = 100.0,
    device=None,
):
    """Convert gsplat-style camera params to the format expected by the
    diff_linear_rasterization CUDA kernel (3DGS column-major convention).

    Args:
        viewmats: ``(1, 4, 4)`` world-to-camera matrix (OpenCV row-major).
        Ks: ``(1, 3, 3)`` intrinsic matrix.

    Returns:
        tanfovx, tanfovy: half-FOV tangents.
        world_view_transform: ``(4, 4)`` column-major W2C (i.e. viewmats[0].T).
        full_proj_transform: ``(4, 4)`` world-to-clip in column-major order.
        campos: ``(3,)`` camera centre in world space.
    """
    vm = viewmats[0].to(device=device, dtype=torch.float32)   # (4, 4) W2C
    K  = Ks[0].to(device=device, dtype=torch.float32)          # (3, 3)

    fx, fy = K[0, 0].item(), K[1, 1].item()
    tanfovx = width  / (2.0 * fx)
    tanfovy = height / (2.0 * fy)

    # 3DGS stores W2C transposed (column-major) so the CUDA kernel can do
    # column-major matrix-vector products.  Must be contiguous — .T alone
    # returns a non-contiguous view which causes illegal memory access in
    # the CUDA kernel.
    world_view_transform = vm.T.contiguous()   # (4, 4)

    # OpenGL-style perspective matrix (matches getProjectionMatrix in the
    # linprim repo's utils/graphics_utils.py).
    top    =  tanfovy * znear
    right  =  tanfovx * znear
    P = torch.zeros(4, 4, device=device, dtype=torch.float32)
    P[0, 0] = 2.0 * znear / (2.0 * right)
    P[1, 1] = 2.0 * znear / (2.0 * top)
    P[3, 2] = 1.0
    P[2, 2] = zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    projection_matrix = P.T.contiguous()   # column-major

    full_proj_transform = (
        world_view_transform.unsqueeze(0)
        .bmm(projection_matrix.unsqueeze(0))
        .squeeze(0)
    )   # bmm output is already contiguous

    # Camera centre in world space.  The 3DGS convention stores world_view_transform
    # as W2C.T, so its inverse is C2W.T; row 3 of C2W.T equals column 3 of C2W,
    # which is the translation (camera centre).
    campos = world_view_transform.inverse()[3, :3]

    return tanfovx, tanfovy, world_view_transform, full_proj_transform, campos


def render_view(
    splats: GaussianSplats,
    viewmats: torch.Tensor,
    Ks: torch.Tensor,
    width: int,
    height: int,
    output_type: str = "3dgs",
    device=None,
    ret_normals: bool = False,
    ret_depth: bool = False,
):
    """Rasterize a ``GaussianSplats`` to RGB, branching on ``output_type``.

    For ``2dgs`` the third column of ``splats.scales`` is ignored; the caller is
    free to pad it with a sentinel value. Returns ``(C, 3, H, W)`` RGB by
    default. If ``ret_normals=True`` (2dgs only) returns ``(rgb, normals)``
    where ``normals`` is ``(C, 3, H, W)``. If ``ret_depth=True`` returns
    ``(rgb, depth)`` where ``depth`` is ``(C, 1, H, W)`` expected z-depth (opacity-normalised).
    """
    if ret_normals and output_type != "2dgs":
        raise ValueError("ret_normals=True is only supported for output_type='2dgs'")

    device = device or splats.means.device

    # ── LinPrim path ──────────────────────────────────────────────────────────
    if output_type == "linprim":
        try:
            from diff_linear_rasterization import (
                LinearRasterizationSettings,
                LinearRasterizer,
            )
        except ImportError as exc:
            raise RuntimeError(
                "diff_linear_rasterization is not installed. "
                "Build it from https://github.com/nicolasvonluetzow/linear-splatting "
                "submodules/diff-linear-rasterization and install with pip."
            ) from exc

        tanfovx, tanfovy, world_view_transform, full_proj_transform, campos = \
            _make_linprim_camera(viewmats, Ks, width, height, device=device)

        # All inputs must be contiguous — the linprim C++ kernel uses raw
        # stride-3/4 flat access and does not call .contiguous() itself.
        # Slices from reconstruct_splats output have strides like (12,1)
        # instead of (3,1), causing the kernel to read garbage data and
        # overflow tile buffers.
        means3D   = splats.means.contiguous()
        # Hard clamp: large dists → too many tiles per primitive → linprim
        # tile-buffer overflow → cudaErrorIllegalAddress.  Cap matches the
        # model head's DIST_MAX so training and reference renders stay in sync.
        _DIST_RENDER_MAX = 7.07  # 2.6 * exp(1.0); matches GaussianHeadLinPrim.DIST_MAX
        dists     = splats.scales.abs().clamp(max=_DIST_RENDER_MAX).expand(-1, 3).contiguous()
        opacities = torch.sigmoid(splats.opacities.flatten()).unsqueeze(-1).contiguous()
        quats     = F.normalize(splats.quats, p=2, dim=-1).contiguous()
        # SH degree 0: (N, 1, 3). The C++ binding infers the SH coefficient
        # count as M = sh.size(1), and the kernel strides idx * M vec3s per
        # point. sh0 is (N, 3), so unsqueeze(1) -> (N, 1, 3) gives M=1 (one
        # coefficient, 3 channels). Using unsqueeze(-1) -> (N, 3, 1) would set
        # M=3, making the kernel read 9 floats/point from a 3-float/point
        # buffer and overrun it (illegal memory access).
        shs = splats.sh0.contiguous().unsqueeze(1)

        subpixel_offset = torch.zeros(height, width, 2,
                                      device=device, dtype=torch.float32)
        raster_settings = LinearRasterizationSettings(
            image_height=height,
            image_width=width,
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            kernel_size=0.0,
            subpixel_offset=subpixel_offset,
            bg=torch.zeros(3, device=device),
            scale_modifier=1.0,
            viewmatrix=world_view_transform,
            projmatrix=full_proj_transform,
            sh_degree=0,
            campos=campos,
            prefiltered=False,
            debug=False,
        )
        screenspace_pts = torch.zeros_like(means3D, requires_grad=True)
        rasterizer = LinearRasterizer(raster_settings=raster_settings)
        rendered_image, _ = rasterizer(
            means3D=means3D,
            means2D=screenspace_pts,
            shs=shs,
            colors_precomp=None,
            opacities=opacities,
            dists=dists,
            rotations=quats,
        )
        # rendered_image: (3, H, W) → (1, 3, H, W)
        return rendered_image.unsqueeze(0)

    # ── 3DGS / 2DGS path ─────────────────────────────────────────────────────
    quats = F.normalize(splats.quats, p=2, dim=-1)
    opacities = torch.sigmoid(splats.opacities.flatten())
    if output_type == "3dgs":
        scales = torch.exp(splats.scales)
    elif output_type == "2dgs":
        scales2d = torch.exp(splats.scales[:, :2])
        z = torch.full(
            (scales2d.shape[0], 1), 1e-4,
            device=scales2d.device, dtype=scales2d.dtype,
        )
        scales = torch.cat([scales2d, z], dim=-1)
    else:
        raise ValueError(f"Unknown output_type: {output_type!r}")

    common = dict(
        means=splats.means,
        quats=quats,
        scales=scales,
        opacities=opacities,
        colors=splats.sh0.unsqueeze(1),
        viewmats=viewmats,
        Ks=Ks,
        width=width,
        height=height,
        sh_degree=0,
        near_plane=0.01,
        backgrounds=torch.zeros(viewmats.shape[0], 3, device=device),
        packed=False,
        absgrad=False,
    )
    render_mode = "RGB+ED" if ret_depth else "RGB"

    if output_type == "3dgs":
        imgs, _, _ = rasterization(
            **common, sparse_grad=False, rasterize_mode="classic", distributed=False,
            render_mode=render_mode,
        )
        if ret_depth:
            return imgs[..., :3].permute(0, 3, 1, 2), imgs[..., 3:4].permute(0, 3, 1, 2)
        return imgs.permute(0, 3, 1, 2)

    out = rasterization_2dgs(**common, render_mode=render_mode)
    rgb = out[0][..., :3].permute(0, 3, 1, 2)
    if ret_depth and not ret_normals:
        return rgb, out[0][..., 3:4].permute(0, 3, 1, 2)
    if not ret_normals:
        return rgb
    # The 2dgs rasterizer returns several auxiliary tensors after RGB; the
    # surface-normal map is the first 3-channel one of them.
    for tensor in out[1:]:
        if not isinstance(tensor, torch.Tensor):
            continue
        t = tensor.permute(0, 3, 1, 2) if tensor.dim() == 4 else tensor
        if t.dim() == 4 and t.shape[1] == 3:
            return rgb, t
    raise RuntimeError("rasterization_2dgs did not return a normal map")


def world_normals_to_camera(normals: torch.Tensor, viewmats: torch.Tensor) -> torch.Tensor:
    """Rotate per-pixel surface normals from world frame into camera frame.

    The 2dgs rasterizer emits ``render_normals`` in world coordinates; the
    depth-derived GT normals from :func:`depth_to_normals` live in the
    camera coordinate frame. To compare them with a normal loss they must
    share a frame, so we apply the world->camera rotation block of each
    ``viewmats`` row to its corresponding normal map.

    Args:
        normals:  ``(C, 3, H, W)`` world-frame normals.
        viewmats: ``(C, 4, 4)`` world->camera transforms (R | t in the top
            3 rows).

    Returns:
        ``(C, 3, H, W)`` camera-frame normals.
    """
    R = viewmats[:, :3, :3]            # (C, 3, 3)
    C, _, H, W = normals.shape
    flat = normals.reshape(C, 3, -1)   # (C, 3, H*W)
    return (R @ flat).reshape(C, 3, H, W)


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

    # Cross product order is `dPdv × dPdu` (not the other way round) so the
    # resulting normal points toward the camera (-Z in camera frame), matching
    # the convention used by `rasterization_2dgs`'s `render_normals` once it's
    # rotated into camera frame by `world_normals_to_camera`.
    normals = torch.cross(
        dPdv.permute(0, 2, 3, 1),
        dPdu.permute(0, 2, 3, 1),
        dim=-1,
    ).permute(0, 3, 1, 2)

    valid_f = (valid > 0).float()                            # (B, 1, H, W)
    normals = normals * valid_f

    norm_len = normals.norm(dim=1, keepdim=True).clamp(min=1e-6)
    normals  = (normals / norm_len) * valid_f                # unit, masked

    return normals