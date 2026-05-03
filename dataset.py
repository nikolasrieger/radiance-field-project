import copy
import json
import os
import sqlite3
import struct
from dataclasses import dataclass
from typing import Optional

import cv2
import imageio.v2 as imageio
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from torch.utils.data import DataLoader, Dataset, Subset


@dataclass
class Cameras:
    R: torch.Tensor                         # (1, 3, 3)
    T: torch.Tensor                         # (1, 3)
    focal_length: torch.Tensor              # (1, 2)
    principal_point: torch.Tensor           # (1, 2)
    colmap_distortion_coeffs: torch.Tensor  # (1, 12)
    image_size: Optional[torch.Tensor] = None  # (1, 2) h,w
    in_ndc: bool = True


@dataclass
class GaussianSplats:
    means: torch.Tensor       # (N, 3)
    sh0: torch.Tensor         # (N, 3)
    opacities: torch.Tensor   # (N,)
    scales: torch.Tensor      # (N, 3)
    quats: torch.Tensor       # (N, 4)  (w, x, y, z)
    shN: Optional[torch.Tensor] = None    # (N, L, 3)
    fg_mask: Optional[torch.Tensor] = None  # (N,)

    def __len__(self):
        return self.means.shape[0]

    @classmethod
    def empty(cls):
        return cls(
            means=torch.empty(0, 3), sh0=torch.empty(0, 3),
            opacities=torch.empty(0), scales=torch.empty(0, 3),
            quats=torch.empty(0, 4),
        )


def _unpack_floats(blob, n):
    return np.array(struct.unpack(f"{n}f", bytes(blob)), dtype=np.float32)


def _unpack_ints(blob, n):
    return np.array(struct.unpack(f"{n}i", bytes(blob)), dtype=np.int32)


def _load_gaussian_splats(splats_dir):
    with open(os.path.join(splats_dir, "meta.json")) as f:
        meta = json.load(f)

    n = meta["means"]["shape"][0]
    meta["sh0"]["shape"] = (n, 3)

    _decompress = {
        "means":     _decomp_png16,
        "scales":    _decomp_png8,
        "quats":     _decomp_png8,
        "opacities": _decomp_png8,
        "sh0":       _decomp_png8,
        "shN":       _decomp_kmeans,
        "fg_mask":   _decomp_npz,
    }

    splats = {}
    for name, m in meta.items():
        val = _decompress[name](splats_dir, name, m)
        if name == "means":
            val = torch.sign(val) * (torch.expm1(torch.abs(val)))
        splats[name] = val.reshape(m["shape"])
    return GaussianSplats(**splats)


def _decomp_png16(d, name, m):
    l_ = imageio.imread(os.path.join(d, f"{name}_l.png"))
    u_ = imageio.imread(os.path.join(d, f"{name}_u.png")).astype(np.uint16)
    norm = ((u_ << 8) + l_).astype(np.float32) / (2**16 - 1)
    lo, hi = torch.tensor(m["mins"]), torch.tensor(m["maxs"])
    return (torch.tensor(norm) * (hi - lo) + lo).to(getattr(torch, m["dtype"]))


def _decomp_png8(d, name, m):
    norm = imageio.imread(os.path.join(d, f"{name}.png")).astype(np.float32) / 255.0
    lo, hi = torch.tensor(m["mins"]), torch.tensor(m["maxs"])
    return (torch.tensor(norm) * (hi - lo) + lo).to(getattr(torch, m["dtype"]))


def _decomp_kmeans(d, name, m):
    npz = np.load(os.path.join(d, f"{name}.npz"))
    lo, hi = torch.tensor(m["mins"]), torch.tensor(m["maxs"])
    centroids = torch.tensor(npz["centroids"].astype(np.float32) / (2 ** m["quantization"] - 1))
    centroids = centroids * (hi - lo) + lo
    labels = torch.tensor(npz["labels"].astype(np.int64))
    return centroids[labels].to(getattr(torch, m["dtype"]))


def _decomp_npz(d, name, m):
    return torch.tensor(np.load(os.path.join(d, f"{name}.npz"))["arr"]).to(getattr(torch, m["dtype"]))


def _ndc_to_px(f_ndc, pp_ndc, size_wh):
    half = size_wh / 2
    s = half.min()
    return f_ndc * s, half - pp_ndc * s


def _px_to_ndc(f_px, pp_px, size_wh):
    half = size_wh / 2
    s = half.min()
    return f_px / s, (half - pp_px) / s


def _undistort(image_rgb, fg_mask, camera, image_size_hw):
    h, w = image_size_hw.tolist()
    size_wh = torch.tensor([w, h], dtype=torch.float32)
    f_px, pp_px = _ndc_to_px(camera.focal_length[0], camera.principal_point[0], size_wh)
    mtx = np.array([
        [f_px[0].item(), 0,              pp_px[0].item()],
        [0,              f_px[1].item(), pp_px[1].item()],
        [0,              0,              1              ],
    ], dtype=np.float64)
    dist = camera.colmap_distortion_coeffs[0].numpy().astype(np.float64)

    new_mtx, _ = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 0.0, (w, h))
    map_x, map_y = cv2.initUndistortRectifyMap(mtx, dist, None, new_mtx, (w, h), cv2.CV_16SC2)

    new_f = torch.tensor([new_mtx[0, 0], new_mtx[1, 1]], dtype=torch.float32)
    new_pp = torch.tensor([new_mtx[0, 2], new_mtx[1, 2]], dtype=torch.float32)
    nf_ndc, npp_ndc = _px_to_ndc(new_f, new_pp, size_wh)
    new_cam = copy.copy(camera)
    new_cam.focal_length = nf_ndc[None]
    new_cam.principal_point = npp_ndc[None]

    def remap(t): 
        arr = t.numpy().transpose(1, 2, 0)
        out = cv2.remap(arr, map_x, map_y, cv2.INTER_LINEAR)
        if out.ndim == 2:
            out = out[..., None]
        return torch.from_numpy(out.transpose(2, 0, 1))

    new_image = remap(image_rgb) if image_rgb is not None else None
    new_mask = remap(fg_mask) if fg_mask is not None else None
    return new_image, new_mask, new_cam


def _get_bbox_from_mask(mask_np, thr=0.4):
    while thr > 0:
        fg = (mask_np > thr).astype(np.float32)
        if fg.sum() > 1:
            break
        thr -= 0.05
    else:
        return 0, 0, mask_np.shape[1], mask_np.shape[0]
    nz_x = np.flatnonzero(fg.sum(0))
    nz_y = np.flatnonzero(fg.sum(1))
    return int(nz_x[0]), int(nz_y[0]), int(nz_x[-1]) - int(nz_x[0]) + 1, int(nz_y[-1]) - int(nz_y[0]) + 1


def _expand_to_xyxy(xywh, context, img_h, img_w):
    x, y, bw, bh = [float(v) for v in xywh]
    x -= bw * context / 2
    y -= bh * context / 2
    bw += bw * context
    bh += bh * context
    x0 = max(0, round(x))
    y0 = max(0, round(y))
    x1 = min(img_w, round(x + bw))
    y1 = min(img_h, round(y + bh))
    return x0, y0, x1, y1


def _crop_camera(camera: Cameras, bbox_xyxy, img_h, img_w):
    x0, y0, x1, y1 = bbox_xyxy
    size_wh = torch.tensor([img_w, img_h], dtype=torch.float32)
    f_px, pp_px = _ndc_to_px(camera.focal_length[0], camera.principal_point[0], size_wh)
    crop_wh = torch.tensor([x1 - x0, y1 - y0], dtype=torch.float32)
    pp_px_cropped = pp_px - torch.tensor([x0, y0], dtype=torch.float32)
    f_ndc, pp_ndc = _px_to_ndc(f_px, pp_px_cropped, crop_wh)
    new_cam = copy.copy(camera)
    new_cam.focal_length = f_ndc[None]
    new_cam.principal_point = pp_ndc[None]
    return new_cam


def _resize(t, tgt_h, tgt_w):
    scale = min(tgt_h / t.shape[-2], tgt_w / t.shape[-1])
    t_r = torch.nn.functional.interpolate(
        t[None].float(), scale_factor=scale, mode="bilinear",
        align_corners=False, recompute_scale_factor=True,
    )[0]
    out = torch.zeros(t.shape[0], tgt_h, tgt_w, dtype=t_r.dtype)
    out[:, :t_r.shape[-2], :t_r.shape[-1]] = t_r
    return out


def _resize_camera(camera: Cameras, src_h, src_w, tgt_h, tgt_w):
    size_wh_src = torch.tensor([src_w, src_h], dtype=torch.float32)
    size_wh_tgt = torch.tensor([tgt_w, tgt_h], dtype=torch.float32)
    f_px, pp_px = _ndc_to_px(camera.focal_length[0], camera.principal_point[0], size_wh_src)
    scale = (size_wh_tgt / size_wh_src).min()
    f_ndc, pp_ndc = _px_to_ndc(f_px * scale, pp_px * scale, size_wh_tgt)
    new_cam = copy.copy(camera)
    new_cam.focal_length = f_ndc[None]
    new_cam.principal_point = pp_ndc[None]
    new_cam.image_size = torch.tensor([[tgt_h, tgt_w]])
    return new_cam


def _apply_alignment(camera, splats, R, T, s):
    new_cam = copy.copy(camera)
    new_cam.R = (R.T @ camera.R[0])[None]
    new_cam.T = (s * (camera.T[0] - T @ camera.R[0]))[None]

    if splats is not None and len(splats) > 0:
        new_means = (splats.means.float() @ R + T) * s
        new_scales = splats.scales.float() + torch.log(s.clamp(min=1e-8))
        q_R = torch.tensor(
            Rotation.from_matrix(R.numpy()).as_quat()[[3, 0, 1, 2]],
            dtype=torch.float32,
        )
        new_quats = _quat_mul(q_R.expand(len(splats), 4), splats.quats.float())
        new_splats = copy.copy(splats)
        new_splats.means = new_means
        new_splats.scales = new_scales
        new_splats.quats = new_quats
    else:
        new_splats = splats

    return new_cam, new_splats


def _quat_mul(q1, q2):
    w1, x1, y1, z1 = q1.unbind(-1)
    w2, x2, y2, z2 = q2.unbind(-1)
    return torch.stack([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ], dim=-1)


class UCO3DDataset(Dataset):
    def __init__(
        self,
        data_root: str = "./data",
        split: str = "train",
        image_height: int = 800,
        image_width: int = 800,
        box_crop_context: float = 0.4,
        gaussian_splats_truncate_background: bool = True,
        normalize_splats: bool = True,
    ):
        self.data_root = data_root
        self.image_height = image_height
        self.image_width = image_width
        self.box_crop_context = box_crop_context
        self.truncate_bg = gaussian_splats_truncate_background
        self.normalize_splats = normalize_splats
        self.index = self._build_index(split)

    def _normalize_gaussian_splats(self, splats):
        means = splats.means  # (N, 3)
        centroid = means.mean(dim=0, keepdim=True)
        means_centered = means - centroid
        position_scale = means_centered.norm(dim=1).max().clamp(min=1e-8)
        means_norm = means_centered / position_scale

        scales_norm = splats.scales - torch.log(position_scale)

        quats = splats.quats
        quats_norm = quats / quats.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        sign = torch.where(quats_norm[:, :1] < 0, -1.0, 1.0)
        quats_norm = quats_norm * sign

        sh0 = splats.sh0
        sh0_mean = sh0.mean(dim=0, keepdim=True)
        sh0_std = sh0.std(dim=0, keepdim=True).clamp(min=1e-8)
        sh0_norm = (sh0 - sh0_mean) / sh0_std

        opacities = splats.opacities.unsqueeze(1) if splats.opacities.dim() == 1 else splats.opacities
        op_mean = opacities.mean(dim=0, keepdim=True)
        op_std = opacities.std(dim=0, keepdim=True).clamp(min=1e-8)
        opacities_norm = (opacities - op_mean) / op_std

        normalized_tensor = torch.cat([
            means_norm,      # 0:3
            sh0_norm,        # 3:6
            opacities_norm,  # 6:7
            scales_norm,     # 7:10
            quats_norm,      # 10:14
        ], dim=-1)

        norm_stats = {
            "centroid": centroid,
            "position_scale": position_scale,
            "sh0_mean": sh0_mean, "sh0_std": sh0_std,
            "op_mean": op_mean,   "op_std": op_std,
        }

        return normalized_tensor, norm_stats

    def _build_index(self, split):
        meta_path = os.path.join(self.data_root, "metadata.sqlite")
        sl_path = os.path.join(self.data_root, "set_lists", "set_lists_all-categories.sqlite")

        conn_sl = sqlite3.connect(sl_path)
        conn_sl.row_factory = sqlite3.Row
        sl_rows = conn_sl.execute(
            "SELECT sequence_name, frame_number FROM set_lists WHERE subset = ?", (split,)
        ).fetchall()
        conn_sl.close()

        keys = {(r["sequence_name"], r["frame_number"]) for r in sl_rows}
        conn = sqlite3.connect(meta_path)
        conn.row_factory = sqlite3.Row
        conn.execute("CREATE TEMP TABLE _k (seq TEXT, frm INTEGER)")
        conn.executemany("INSERT INTO _k VALUES (?,?)", list(keys))
        rows = conn.execute("""
            SELECT
                fa.sequence_name,  fa.frame_number,  fa.frame_timestamp,
                fa._viewpoint_R,   fa._viewpoint_T,
                fa._viewpoint_focal_length,       fa._viewpoint_principal_point,
                fa._viewpoint_colmap_distortion_coeffs,
                fa._image_size,    fa._mask_bounding_box_xywh,
                sa._video_path,    sa._mask_video_path,
                sa._gaussian_splats_dir,
                sa._alignment_R,   sa._alignment_T,  sa._alignment_scale
            FROM frame_annots fa
            INNER JOIN _k        ON fa.sequence_name = _k.seq
                                AND fa.frame_number  = _k.frm
            INNER JOIN sequence_annots sa ON fa.sequence_name = sa.sequence_name
        """).fetchall()
        conn.close()

        index = []
        for r in rows:
            R   = _unpack_floats(r["_viewpoint_R"], 9)
            T   = _unpack_floats(r["_viewpoint_T"], 3)
            fl  = _unpack_floats(r["_viewpoint_focal_length"], 2)
            pp  = _unpack_floats(r["_viewpoint_principal_point"], 2)
            if any(v is None for v in (R, T, fl, pp)):
                continue
            dist = _unpack_floats(r["_viewpoint_colmap_distortion_coeffs"], 12)
            hw   = _unpack_ints(r["_image_size"], 2)
            bbox = _unpack_floats(r["_mask_bounding_box_xywh"], 4)
            aR   = _unpack_floats(r["_alignment_R"], 9)
            aT   = _unpack_floats(r["_alignment_T"], 3)
            aS   = float(r["_alignment_scale"]) if r["_alignment_scale"] is not None else 1.0
            index.append({
                "sequence_name":   r["sequence_name"],
                "frame_number":    r["frame_number"],
                "frame_timestamp": float(r["frame_timestamp"] or 0.0),
                "R": R.reshape(3, 3), "T": T, "fl": fl, "pp": pp,
                "dist": dist if dist is not None else np.zeros(12, np.float32),
                "image_size_hw": hw,
                "bbox_xywh": bbox,
                "video_path":      r["_video_path"],
                "mask_video_path": r["_mask_video_path"],
                "gaussian_splats_dir": r["_gaussian_splats_dir"],
                "align_R": aR.reshape(3, 3) if aR is not None else None,
                "align_T": aT,
                "align_s": aS,
            })
        return index

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        fd = self.index[idx]

        image_rgb = self._frame_from_video(fd["video_path"], fd["frame_timestamp"], gray=False)
        fg_mask   = self._frame_from_video(fd["mask_video_path"], fd["frame_timestamp"], gray=True)
        image_size_hw = torch.tensor(fd["image_size_hw"][:2], dtype=torch.long)

        camera = Cameras(
            R=torch.tensor(fd["R"],    dtype=torch.float32)[None],
            T=torch.tensor(fd["T"],    dtype=torch.float32)[None],
            focal_length=torch.tensor(fd["fl"], dtype=torch.float32)[None],
            principal_point=torch.tensor(fd["pp"], dtype=torch.float32)[None],
            colmap_distortion_coeffs=torch.tensor(fd["dist"], dtype=torch.float32)[None],
            image_size=image_size_hw[None],
        )

        image_rgb, fg_mask, camera = _undistort(image_rgb, fg_mask, camera, image_size_hw)

        h, w = image_rgb.shape[-2], image_rgb.shape[-1]
        if fd["bbox_xywh"] is not None:
            bbox_xywh = fd["bbox_xywh"]
        elif fg_mask is not None:
            bbox_xywh = _get_bbox_from_mask(fg_mask[0].numpy())
        else:
            bbox_xywh = None

        if bbox_xywh is not None:
            x0, y0, x1, y1 = _expand_to_xyxy(bbox_xywh, self.box_crop_context, h, w)
            image_rgb = image_rgb[:, y0:y1, x0:x1]
            if fg_mask is not None:
                fg_mask = fg_mask[:, y0:y1, x0:x1]
            camera = _crop_camera(camera, (x0, y0, x1, y1), h, w)
            crop_h, crop_w = y1 - y0, x1 - x0
        else:
            crop_h, crop_w = h, w

        image_rgb = _resize(image_rgb, self.image_height, self.image_width)
        if fg_mask is not None:
            fg_mask = _resize(fg_mask, self.image_height, self.image_width)
        camera = _resize_camera(camera, crop_h, crop_w, self.image_height, self.image_width)

        splats = self._load_splats(fd["gaussian_splats_dir"])
        if self.truncate_bg and splats.fg_mask is not None:
            fg = splats.fg_mask.reshape(-1) > 0.5
            splats = GaussianSplats(
                means=splats.means[fg], sh0=splats.sh0[fg],
                opacities=splats.opacities[fg], scales=splats.scales[fg],
                quats=splats.quats[fg],
                shN=splats.shN[fg] if splats.shN is not None else None,
            )

        if fd["align_R"] is not None and fd["align_T"] is not None:
            R = torch.tensor(fd["align_R"], dtype=torch.float32)
            T = torch.tensor(fd["align_T"], dtype=torch.float32)
            s = torch.tensor(fd["align_s"], dtype=torch.float32)
            camera, splats = _apply_alignment(camera, splats, R, T, s)
        
        data = {
            "image_rgb":                    image_rgb,
            "fg_probability":               fg_mask,
            "camera":                       camera,
            "sequence_gaussian_splats":     splats,
            "sequence_name":                fd["sequence_name"],
            "frame_number":                 fd["frame_number"],
        }

        if self.normalize_splats:
            norm_tensor, norm_stats = self._normalize_gaussian_splats(splats)
            data["splat_tensor"] = norm_tensor
            data["norm_stats"] = norm_stats
        else:
            data["sequence_gaussian_splats"] = splats

        return data

    def _frame_from_video(self, rel_path, timestamp_sec, gray=False):
        path = os.path.join(self.data_root, rel_path)
        cap = cv2.VideoCapture(path)
        cap.set(cv2.CAP_PROP_POS_MSEC, timestamp_sec * 1000)
        ret, frame = cap.read()
        cap.release()
        if gray:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            return torch.from_numpy(frame.astype(np.float32) / 255.0)[None]
        frame = frame[..., ::-1].astype(np.float32) / 255.0
        return torch.from_numpy(np.ascontiguousarray(frame.transpose(2, 0, 1)))

    def _load_splats(self, rel_dir):
        path = os.path.join(self.data_root, rel_dir)
        return _load_gaussian_splats(path)


def get_dataloader(data_root, batch_size, split = "train", num_samples = None):
    dataset = UCO3DDataset(data_root=data_root, split=split, normalize_splats=True)
    subset_indices = range(num_samples) if num_samples is not None else range(len(dataset))
    small_dataset = Subset(dataset, subset_indices)
    return DataLoader(
        small_dataset,
        batch_size=batch_size,
        shuffle=(split == "train"),
        num_workers=0,
        collate_fn=lambda batch: batch,
    )


if __name__ == "__main__":
    loader = get_dataloader("./data", batch_size=1)
    (sample,) = next(iter(loader))
    print("Batch loaded!")
    print(f"image_rgb:            {sample['image_rgb'].shape}")
    print(f"fg_probability:       {sample['fg_probability'].shape}")
    print(f"splat_tensor:         {sample['splat_tensor'].shape}")