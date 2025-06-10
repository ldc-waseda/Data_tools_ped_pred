from util import *
from data_loader import ETHLoader, SDDLoader
import pandas as pd
import streamlit as st
import io
import imageio
import cv2
import torch
import os
import matplotlib.pyplot as plt
import numpy as np

script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)

SDD_PATH = os.path.join(project_dir, 'sdd_dataset')
SDD_FILE = [
    'bookstore',
    'coupa',
    'deathCircle',
    'gates',
    'hyang',
    'little',
    'nexus',
    'quad',
]
ETH_PATH = os.path.join(project_dir, 'eth_dataset')
ETH_FILE = [
    'ETH',
    'HOTEL',
    'ZARA01',
    'ZARA02',
    'STUDENT',
]

sdd_datasets_txt_list, sdd_datasets_video_list = load_sdd_paths(SDD_PATH, SDD_FILE)
eth_datasets_txt_list, eth_datasets_video_list = load_eth_paths(ETH_PATH, ETH_FILE)

merged_txt_paths   = {**sdd_datasets_txt_list,  **eth_datasets_txt_list}
merged_video_paths = {**sdd_datasets_video_list,  **eth_datasets_video_list}
ALL_SCENES = ETH_FILE + SDD_FILE

st.sidebar.header('场景与段选择')

# 1) 选择场景
scene = st.sidebar.selectbox('请选择场景', ALL_SCENES)

# 从合并字典中取出该场景的所有 txt / video 列表
txt_list = merged_txt_paths.get(scene, [])
vid_list = merged_video_paths.get(scene, [])

if not txt_list:
    st.sidebar.warning(f"场景 {scene} 未找到任何 .txt 文件")
if not vid_list:
    st.sidebar.warning(f"场景 {scene} 未找到任何视频文件")

# 2) 如果是 SDD 场景，再选择具体哪个 segment
if scene in SDD_FILE:
    segments = sorted({os.path.basename(os.path.dirname(p)) for p in txt_list})
    segment = st.sidebar.selectbox('请选择 Segment', segments)

    # 过滤出这个 segment 下的文件
    txt_file = next(p for p in txt_list if os.path.basename(os.path.dirname(p)) == segment)
    vid_file = next(p for p in vid_list if os.path.basename(os.path.dirname(p)) == segment)

else:
    # ETH 场景：直接让用户从文件列表里选一个
    txt_file = st.sidebar.selectbox('标注文件 (.txt)', txt_list)
    vid_file = st.sidebar.selectbox('视频文件 (.avi/.mov)', vid_list)

# 3) 加载数据
if scene in SDD_FILE:
    data_loader = SDDLoader(txt_file)
else:
    data_loader = ETHLoader(txt_file)

obs = torch.tensor(data_loader.target_obs, dtype=torch.float32)
obs.tag = data_loader.xy_tag
num_samples = obs.shape[0]
st.write(
    f"### 当前场景：**{scene}**"
    + (f" / Segment **{segment}**" if scene in SDD_FILE else "")
    + (f"  Total NUM **{num_samples}**")
)
if num_samples == 0:
    st.warning("该场景无可用样本")
    st.stop()

# 4.1) 在侧边栏选择当前样本索引
sample_idx = st.sidebar.number_input(
    "选择样本索引",
    min_value=0,
    max_value=num_samples - 1,
    value=0,
    step=1,
    format="%d"
)

# 4.2) 提取该样本的数据：frames, id, xy
sample = obs[sample_idx]            # shape [frames, feat_dim]
frames = sample[:, 0].long().numpy()
agent_id = int(sample[0, 1].item())
if obs.tag == 0:
    traj = sample[:, [3, 2]].numpy()   # shape [frames, 2]
else:
    traj = sample[:, [2, 3]].numpy()   # shape [frames, 2]

# —————————————————————————————————————————————————————
# 新增：计算观测序列与预测序列的平均速度
# —————————————————————————————————————————————————————
obs_len = 8
pred_len = 12

total_len = traj.shape[0]
use_obs_len = min(obs_len, total_len)
use_pred_len = min(pred_len, total_len - use_obs_len)

obs_traj = traj[:use_obs_len]
pred_traj = traj[use_obs_len:use_obs_len + use_pred_len]

def compute_avg_speed(sequence):
    if sequence.shape[0] < 2:
        return 0.0
    diffs = sequence[1:] - sequence[:-1]
    dists = np.linalg.norm(diffs, axis=1)
    return float(dists.mean())

avg_speed_obs  = compute_avg_speed(obs_traj)
avg_speed_pred = compute_avg_speed(pred_traj)

# 将数值转成带两位小数的字符串
obs_speed_str  = f"{avg_speed_obs:.2f} px/frame"
pred_speed_str = f"{avg_speed_pred:.2f} px/frame"
# —————————————————————————————————————————————————————

# —————————————————————————————————————————————————————
# 修改：将“速度”栏改为自动填充观测/预测段平均速度，
#        只保留“程度”和“方向”的手动选择
# —————————————————————————————————————————————————————

# 定义“程度”和“方向”关键词
keywords2 = ["slight", "extreme"]
keywords3 = [
    "forward", "left-forward", "right-forward",
    "left", "right", "back",
    "left-back", "right-back"
]

default_index2 = ([""] + keywords2).index("slight") if "slight" in keywords2 else 0
default_index3 = ([""] + keywords3).index("forward") if "forward" in keywords3 else 0

# 单一 annotation key
annotation_key = f"ann_{scene}_{sample_idx}"
if annotation_key not in st.session_state:
    st.session_state[annotation_key] = ""

# 左侧：观测部分（显示自动速度，手动程度与方向）
st.sidebar.markdown("## 观测序列标签（red）")
st.sidebar.write(f"**自动速度：** {obs_speed_str} (像素/帧)")

obs_degree    = st.sidebar.selectbox("1️⃣ 程度", [""] + keywords2, key=f"{annotation_key}_obs_degree", index=default_index2)
obs_direction = st.sidebar.selectbox("2️⃣ 方向", [""] + keywords3, key=f"{annotation_key}_obs_direction", index=default_index3)

# 左侧：预测部分
st.sidebar.markdown("---")
st.sidebar.markdown("## 预测序列标签（green）")
st.sidebar.write(f"**自动速度：** {pred_speed_str} (像素/帧)")

pred_degree    = st.sidebar.selectbox("1️⃣ 程度", [""] + keywords2, key=f"{annotation_key}_pred_degree", index=default_index2)
pred_direction = st.sidebar.selectbox("2️⃣ 方向", [""] + keywords3, key=f"{annotation_key}_pred_direction", index=default_index3)
pred_confidence = st.sidebar.selectbox(
    "3️⃣ 置信度 (yes/no)",
    ["", "yes", "no"],
    key=f"{annotation_key}_pred_confidence"
)
st.sidebar.markdown("---")

# 只要四项都非空，就自动拼接
if (
    obs_degree and obs_direction
    and pred_degree and pred_direction
):
    filled = (
        f"a person go {{{obs_speed_str}}} {{{obs_degree}}} {{{obs_direction}}} "
        f"then {{{pred_speed_str}}} {{{pred_degree}}} {{{pred_direction}}}"
    )
    if st.session_state[annotation_key] != filled:
        st.session_state[annotation_key] = filled

# —————————————————————————————————————————————————————
# 结束“修改部分”
# —————————————————————————————————————————————————————

# 4.3) 展示样本基本信息
st.write(f"**样本 {sample_idx}** — Agent ID: {agent_id} — 长度: {len(frames)} 帧")
st.write(f"- 观测段平均速度 (像素/帧)：**{avg_speed_obs:.2f}**")
st.write(f"- 预测段平均速度 (像素/帧)：**{avg_speed_pred:.2f}**")

# 4.4) 读取起始帧并作背景图
st.subheader("Ped View")
cap = cv2.VideoCapture(vid_file)
cap.set(cv2.CAP_PROP_POS_FRAMES, int(frames[0]))
ret, frame = cap.read()
cap.release()

if ret:
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(frame_rgb)
    traj_int = traj.astype(int)

    # 绘制观测段（红色）和预测段（绿色）
    first_pts = traj_int[:use_obs_len]
    ax.plot(
        first_pts[:, 0], first_pts[:, 1],
        '-o', color='red',
        markersize=4, linewidth=2,
    )
    second_pts = traj_int[use_obs_len:use_obs_len + use_pred_len]
    ax.plot(
        second_pts[:, 0], second_pts[:, 1],
        '-o', color='green',
        markersize=4, linewidth=2,
    )

    # 在图上标注平均速度
    ax.text(
        5, 15,
        f"Obs avg: {avg_speed_obs:.2f}",
        color='white', fontsize=10,
        bbox=dict(facecolor='red', alpha=0.5, pad=2)
    )
    ax.text(
        5, 55,
        f"Pred avg: {avg_speed_pred:.2f}",
        color='white', fontsize=10,
        bbox=dict(facecolor='green', alpha=0.5, pad=2)
    )

    ax.set_title(f"Global View of Ped: {agent_id} from Frame: {frames[0]}")
    ax.axis('off')
    # ax.legend(loc='lower right', fontsize=8)
    st.pyplot(fig)
else:
    st.warning(f"无法读取帧 {frames[0]}，无法可视化轨迹")

# 4.5) 读取所有帧，叠加轨迹，生成 GIF
st.subheader("GIF 可视化")
gif_container = st.empty()

def make_gif_bytes(images):
    buf = io.BytesIO()
    with imageio.get_writer(buf, format='GIF', mode='I', fps=5) as writer:
        for img in images:
            writer.append_data(img)
    buf.seek(0)
    return buf.read()

cap = cv2.VideoCapture(vid_file)
traj_int = traj.astype(int)
images_rgb = []
for idx, t in enumerate(frames):
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(t))
    ret, frame = cap.read()
    if not ret:
        continue
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    p = tuple(traj_int[idx])
    cv2.circle(frame_rgb, p, radius=15, color=(0, 0, 255), thickness=-1)
    images_rgb.append(frame_rgb)
cap.release()

# 初次生成并展示 GIF
gif_bytes = make_gif_bytes(images_rgb)
gif_container.image(gif_bytes, caption="当前点动画 (GIF)", use_container_width=True)

if st.button("Replay"):
    gif_bytes = make_gif_bytes(images_rgb)
    gif_container.image(gif_bytes, caption="当前点动画 (GIF)", use_container_width=True)

# 4.6) 在主区域展示单句注释，用户可手动微调
st.markdown("### 注释 (Observation → Prediction)")
st.text_area(
    "自动拼接并可编辑：",
    value=st.session_state[annotation_key],
    key=annotation_key,
    height=80
)

# 4.7) 保存按钮
if st.sidebar.button("保存样本数据", key=f"save_npz_{scene}_{sample_idx}"):
    # 1) 计算 sec_id
    if scene in SDD_FILE:
        sec_id = os.path.basename(os.path.dirname(txt_file))
    else:
        sec_id = scene
    pred_confidence = st.session_state.get(f"{annotation_key}_pred_confidence", "")
    print(pred_confidence)
    # 2) 准备要保存的字典
    data_dict = {
        "id":          np.int32(agent_id),
        "start_frame": np.int32(frames[0]),
        "total_seq":   np.int32(len(frames)),
        "traj":        traj.astype(np.float32),
        "start_img":   images_rgb[0].astype(np.uint8),
        "annotation":  np.array(st.session_state[annotation_key], dtype="U"),
        "avg_speed_obs":  np.float32(avg_speed_obs),
        "avg_speed_pred": np.float32(avg_speed_pred),
        "pred_confidence": np.array(pred_confidence, dtype="U")
    }
    # 3) 文件名 & 保存
    dataset_name = "SDD" if scene in SDD_FILE else "ETH"
    filename = f"{dataset_name}_{scene}_{sec_id}_{agent_id}.npz"
    os.makedirs("annotations_npz", exist_ok=True)
    save_path = os.path.join("annotations_npz", filename)
    np.savez_compressed(save_path, **data_dict)

    st.sidebar.success(f"已保存 → {save_path}")
