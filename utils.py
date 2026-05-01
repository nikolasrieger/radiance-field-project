import numpy as np

def read_ply_points(path, max_points=1024):
    import struct as _struct

    with open(path, "rb") as f:
        header = b""
        num_vertices = 0
        is_binary = False
        while True:
            line = f.readline()
            header += line
            decoded = line.decode("ascii", errors="ignore").strip()
            if decoded.startswith("element vertex"):
                num_vertices = int(decoded.split()[-1])
            if decoded.startswith("format binary"):
                is_binary = True
            if decoded == "end_header":
                break

        xyz_list = []
        rgb_list = []

        if is_binary:
            for _ in range(min(num_vertices, max_points)):
                data = f.read(24)
                if len(data) < 24:
                    break
                vals = _struct.unpack("<6f", data)
                xyz_list.append(vals[:3])
                rgb_list.append(vals[3:6])
        else:
            for _ in range(min(num_vertices, max_points)):
                parts = f.readline().decode().strip().split()
                if len(parts) < 3:
                    continue
                xyz_list.append([float(parts[0]), float(parts[1]), float(parts[2])])
                rgb_list.append(
                    [float(parts[3]), float(parts[4]), float(parts[5])]
                    if len(parts) >= 6
                    else [0.0, 0.0, 0.0]
                )

    xyz = np.array(xyz_list, dtype=np.float32) if xyz_list else np.zeros((0, 3), dtype=np.float32)
    rgb = np.array(rgb_list, dtype=np.float32) if rgb_list else np.zeros((0, 3), dtype=np.float32)
    return xyz, rgb