import os
import sqlite3
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from utils import read_ply_points


class GaussianSceneDataset(Dataset):
    def __init__(self, data_dir, split="train", img_size=(512, 512), max_points=1024):
        super().__init__()
        self.data_dir = data_dir
        self.split = split
        self.img_size = img_size
        self.max_points = max_points

        self.metadata_path = os.path.join(data_dir, "metadata.sqlite")
        self.set_list_path = os.path.join(data_dir, "set_lists", "set_lists_all-categories.sqlite")

        self.frames = self._load_frame_metadata()

    def _load_frame_metadata(self):
        conn_sl = sqlite3.connect(self.set_list_path)
        sl_df = self._query(conn_sl, "SELECT sequence_name, frame_number FROM set_lists WHERE subset = ?", (self.split,))
        conn_sl.close()

        if not sl_df:
            raise RuntimeError(f"No frames found for split '{self.split}' in {self.set_list_path}")

        split_keys = {(r["sequence_name"], r["frame_number"]) for r in sl_df}
        conn_meta = sqlite3.connect(self.metadata_path)

        conn_meta.execute("CREATE TEMP TABLE _split_keys (sequence_name TEXT, frame_number INTEGER)")
        conn_meta.executemany("INSERT INTO _split_keys VALUES (?, ?)", list(split_keys))

        rows = self._query(
            conn_meta,
            """
            SELECT fa.sequence_name,
                fa.frame_number,
                fa."_image_path"                         AS image_path,
                fa."_mask_path"                          AS mask_path,
                fa."_depth_path"                         AS depth_path,
                fa."_viewpoint_R"                        AS R,
                fa."_viewpoint_T"                        AS T,
                fa."_viewpoint_focal_length"             AS focal_length,
                fa."_viewpoint_principal_point"          AS principal_point
            FROM frame_annots fa
            INNER JOIN _split_keys sk
                ON fa.sequence_name = sk.sequence_name
            AND fa.frame_number = sk.frame_number
            """,
        )
        conn_meta.close()

        frames = []
        for r in rows:
            R_raw  = r.get("R")
            T_raw  = r.get("T")
            fl_raw = r.get("focal_length")
            pp_raw = r.get("principal_point")

            if any(v is None for v in (R_raw, T_raw, fl_raw, pp_raw)):
                continue

            R  = np.array(self._parse_vec(R_raw),  dtype=np.float32).reshape(3, 3)
            T  = np.array(self._parse_vec(T_raw),  dtype=np.float32)
            fl = np.array(self._parse_vec(fl_raw), dtype=np.float32)  # [fx, fy]
            pp = np.array(self._parse_vec(pp_raw), dtype=np.float32)  # [cx, cy]

            image_rel = r.get("image_path") or ""
            seq_dir   = os.path.join(self.data_dir, os.path.dirname(image_rel)) if image_rel else ""

            frames.append({
                "sequence_name":   r["sequence_name"],
                "frame_number":    r["frame_number"],
                "seq_dir":         seq_dir,
                "rgb_video_path":  os.path.join(self.data_dir, image_rel) if image_rel else "",
                "mask_video_path": os.path.join(self.data_dir, r["mask_path"]) if r.get("mask_path") else "",
                "sparse_pc_path":  os.path.join(seq_dir, "sparse_point_cloud.ply") if seq_dir else "",
                "R":         R,
                "T":         T,
                "focal":     fl,
                "principal": pp,
            })

        return frames

    @staticmethod
    def _parse_vec(value):
        arr = np.frombuffer(bytes(value), dtype=np.float32) 
        return arr.tolist()

    @staticmethod
    def _query(conn, sql, params=()):
        conn.row_factory = sqlite3.Row
        cur = conn.execute(sql, params)
        return [dict(row) for row in cur.fetchall()]

    def _extract_frame_from_video(self, video_path, frame_idx):
        if not video_path or not os.path.isfile(video_path):
            return np.zeros((self.img_size[1], self.img_size[0], 3), dtype=np.uint8)

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return np.zeros((self.img_size[1], self.img_size[0], 3), dtype=np.uint8)

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        cap.release()

        if not ret or frame is None:
            return np.zeros((self.img_size[1], self.img_size[0], 3), dtype=np.uint8)

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, self.img_size, interpolation=cv2.INTER_AREA)
        return frame

    def _load_point_cloud(self, path):
        if not path or not os.path.isfile(path):
            return (
                torch.zeros((self.max_points, 3), dtype=torch.float32),
                torch.zeros((self.max_points, 3), dtype=torch.float32),
            )

        xyz, rgb = read_ply_points(path, max_points=self.max_points)
        n = xyz.shape[0]

        if n < self.max_points:
            pad = self.max_points - n
            xyz = np.concatenate([xyz, np.zeros((pad, 3), dtype=np.float32)])
            rgb = np.concatenate([rgb, np.zeros((pad, 3), dtype=np.float32)])
        else:
            xyz = xyz[: self.max_points]
            rgb = rgb[: self.max_points]

        return (torch.from_numpy(xyz), torch.from_numpy(rgb))

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        fd = self.frames[idx]

        img_np = self._extract_frame_from_video(fd["rgb_video_path"], fd["frame_number"])
        
        img = torch.from_numpy(img_np).float().permute(2, 0, 1) / 255.0

        R = torch.tensor(fd["R"], dtype=torch.float32)
        T = torch.tensor(fd["T"], dtype=torch.float32)
        xyz, features = self._load_point_cloud(fd["sparse_pc_path"])

        return {
            "image": img,
            "R": R,
            "T": T,
            "fx": fd["focal"][0],
            "fy": fd["focal"][1],
            "cx": fd["principal"][0],
            "cy": fd["principal"][1],
            "width": float(self.img_size[0]),
            "height": float(self.img_size[1]),
            "xyz": xyz,
            "features": features,
            "frame_idx": idx,
        }

    def get_sequence_name(self, idx):
        return self.frames[idx]["sequence_name"]


def get_dataloader(data_dir, batch_size=4, num_workers=4, split="train"):
    dataset = GaussianSceneDataset(data_dir, split=split)
    return DataLoader(dataset, batch_size=batch_size, shuffle=(split == "train"), num_workers=num_workers)


if __name__ == "__main__":
    loader = get_dataloader("./data")
    batch = next(iter(loader))
    print("Batch loaded!")
    print(f"Image shape: {batch['image'].shape}")
    print(f"Point cloud shape: {batch['xyz'].shape}")
