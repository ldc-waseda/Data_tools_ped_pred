import os
import re
import numpy as np

def load_annotations(rel_folder="annotations_npz"):
    # 获取当前脚本所在目录
    base_dir = os.path.dirname(os.path.abspath(__file__))
    folder_path = os.path.join(base_dir, rel_folder)

    if not os.path.isdir(folder_path):
        raise FileNotFoundError(f"找不到注释目录：{folder_path}")

    pattern = re.compile(
        r"^(?P<dataset>[^_]+)_(?P<scene>[^_]+)_(?P<sec>[^_]+)_(?P<agent>[^_]+)\.npz$"
    )
    records = []
    for fname in os.listdir(folder_path):
        if not fname.endswith(".npz"):
            continue
        m = pattern.match(fname)
        if not m:
            continue
        info = m.groupdict()
        full_path = os.path.join(folder_path, fname)
        npz = np.load(full_path, allow_pickle=True)
        records.append({
            **info,
            "id":           int(npz["id"]),
            "start_frame":  int(npz["start_frame"]),
            "total_seq":    int(npz["total_seq"]),
            "traj":         npz["traj"],
            "start_img":    npz["start_img"],
            "annotation":   npz["annotation"]
        })
    return records

if __name__ == "__main__":
    ann_list = load_annotations()  # 默认去同级的 annotations_npz
    print(f"共加载 {len(ann_list)} 条记录")
