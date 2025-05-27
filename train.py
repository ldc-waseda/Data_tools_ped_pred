import os
import re
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader

def load_annotations(rel_folder="annotations_npz"):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    folder_path = os.path.join(base_dir, rel_folder)
    if not os.path.isdir(folder_path):
        raise FileNotFoundError(f"找不到注释目录：{folder_path}")

    pattern = re.compile(r"^(?P<dataset>[^_]+)_(?P<scene>[^_]+)_(?P<sec>[^_]+)_(?P<agent>[^_]+)\.npz$")
    records = []
    for fname in os.listdir(folder_path):
        if not fname.endswith(".npz"):
            continue
        m = pattern.match(fname)
        if not m:
            continue
        info = m.groupdict()
        data = np.load(os.path.join(folder_path, fname), allow_pickle=True)
        records.append({
            **info,
            "id":           int(data["id"]),
            "start_frame":  int(data["start_frame"]),
            "total_seq":    int(data["total_seq"]),
            "traj":         data["traj"],        # (T,2)
            "start_img":    data["start_img"],   # (H,W,3)
            "annotation":   data["annotation"]   # (T,) unicode array
        })
        # print({
        #     **info,
        #     "id":           int(data["id"]),
        #     "start_frame":  int(data["start_frame"]),
        #     "total_seq":    int(data["total_seq"]),
        #     "traj":         data["traj"],        # (T,2)
        #     "start_img":    data["start_img"],   # (H,W,3)
        #     "annotation":   data["annotation"]   # (T,) unicode array
        # })
    return records

class AnnotationDataset(Dataset):
    def __init__(self, folder_path, img_transform=None, traj_transform=None):
        self.records = load_annotations(folder_path)
        self.img_tf  = img_transform
        self.traj_tf = traj_transform

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec     = self.records[idx]
        img_np  = rec["start_img"]                     # H,W,3 uint8
        traj_np = rec["traj"]                          # T,2 float32
        ann_np  = rec["annotation"]                    # (T,) array of str

        # 图像 -> Tensor [3,H,W]
        img = torch.from_numpy(img_np).permute(2,0,1).float().div(255.0)
        if self.img_tf:
            img = self.img_tf(img)

        # 轨迹 -> Tensor [T,2]
        traj = torch.from_numpy(traj_np).float()
        if self.traj_tf:
            traj = self.traj_tf(traj)

        annotation = ann_np.tolist()  # list of str

        return {
            "dataset":     rec["dataset"],
            "scene":       rec["scene"],
            "sec_id":      rec["sec"],
            "agent_id":    rec["agent"],
            "image":       img,             # [3,H,W]
            "traj":        traj,            # [T,2]
            "annotation":  annotation,      # list[str]
            "start_frame": rec["start_frame"],
            "total_seq":   rec["total_seq"]
        }

def collate_fn(batch):
    # batch: list of sample dicts
    # print("DEBUG sample keys:", batch[0].keys())  
    images = torch.stack([b["image"] for b in batch], dim=0)  # [B,3,H,W]
    trajs  = [b["traj"] for b in batch]
    lengths = torch.tensor([t.shape[0] for t in trajs], dtype=torch.long)

    # pad trajs to (B, T_max, 2)
    padded_trajs = pad_sequence(trajs, batch_first=True, padding_value=0.0)

    # mask indicating valid steps
    max_len = padded_trajs.size(1)
    mask = (torch.arange(max_len)[None, :] < lengths[:, None])

    # collect annotations as list of list[str]
    annotations = [b["annotation"] for b in batch]

    # collect metadata
    meta = {
        "dataset":     [b["dataset"] for b in batch],
        "scene":       [b["scene"] for b in batch],
        "sec_id":      [b["sec_id"] for b in batch],
        "agent_id":    [b["agent_id"] for b in batch],
        "start_frame": torch.tensor([b["start_frame"] for b in batch], dtype=torch.long),
        "total_seq":   torch.tensor([b["total_seq"] for b in batch],   dtype=torch.long)
    }

    return {
        "image":       images,          # Tensor [B,3,H,W]
        "traj":        padded_trajs,    # Tensor [B,T_max,2]
        "mask":        mask,            # BoolTensor [B,T_max]
        "annotation":  annotations,     # list of list[str]
        **meta
    }

if __name__ == "__main__":
    from torchvision import transforms

    ds = AnnotationDataset(
        folder_path="./annotations_npz",
        img_transform=transforms.Resize((224,224))
    )
    loader = DataLoader(
        ds,
        batch_size=8,
        shuffle=True,
        num_workers=4,
        collate_fn=collate_fn
    )

    # for batch in loader:
    #     print(batch)
