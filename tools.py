from util import *
from data_loader import ETHLoader, SDDLoader
import pandas as pd
import streamlit as st
import io
import imageio
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
    # 从路径中提取所有 segment 名：annotations/<scene>/<segment>/annotations.txt
    segments = sorted({ os.path.basename(os.path.dirname(p)) for p in txt_list })
    segment = st.sidebar.selectbox('请选择 Segment', segments)

    # 过滤出这个 segment 下的文件
    txt_file = next(p for p in txt_list if os.path.basename(os.path.dirname(p)) == segment)
    vid_file = next(p for p in vid_list if os.path.basename(os.path.dirname(p)) == segment)

else:
    # ETH 场景：直接让用户从文件列表里选一个
    txt_file = st.sidebar.selectbox('标注文件 (.txt)', txt_list)
    vid_file = st.sidebar.selectbox('视频文件 (.avi/.mov)', vid_list)


# 3) 在主区域展示所选文件
# st.write(f"### 当前场景：**{scene}**" + (f" / Segment **{segment}**" if scene in SDD_FILE else ""))


# 4) 加载数据并可视化样本
# 根据场景类型选择对应的 Loader
if scene in SDD_FILE:
    data_loader = SDDLoader(txt_file)
else:
    data_loader = ETHLoader(txt_file)

# data_loader.target_obs: List 或 ndarray，shape = [NumSamples, num_frames, feat_dim]
# 转为 tensor，方便索引
obs = torch.tensor(data_loader.target_obs, dtype=torch.float32)
obs.tag = data_loader.xy_tag
num_samples = obs.shape[0]
st.write(f"### 当前场景：**{scene}**" + (f" / Segment **{segment}**" if scene in SDD_FILE else "")+ (f" Total NUM **{num_samples}**"))
if num_samples == 0:
    st.warning("该场景无可用样本")
else:
    # 4.1 在侧边栏选择当前样本索引
    sample_idx = st.sidebar.number_input(
    "选择样本索引",
    min_value=0,
    max_value=num_samples - 1,
    value=0,
    step=1,
    format="%d"       # 整数格式
)
    # 提取该样本的数据：frames, id, xy
    sample = obs[sample_idx]            # shape [frames, feat_dim]
    frames = sample[:, 0].long().numpy()
    agent_id = int(sample[0, 1].item())
    if obs.tag == 0:
        traj = sample[:, [3, 2]].numpy()   # shape [frames, 2]
    else:
        traj = sample[:, [2, 3]].numpy()   # shape [frames, 2]

    # 4.2 展示样本基本信息
    st.write(f"**样本 {sample_idx}** — Agent ID: {agent_id} — 长度: {len(frames)} 帧")

    # 4.3 读取起始帧并作背景图
    st.subheader("Ped View")  # 新增标题
    cap = cv2.VideoCapture(vid_file)
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frames[0]))
    ret, frame = cap.read()
    cap.release()

    if ret:
        # BGR -> RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # 4.4 在该帧上绘制轨迹
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.imshow(frame_rgb)                      # 先把帧当背景图
        traj_int = traj.astype(int)
        first_pts = traj_int[:8]
        ax.plot(
            first_pts[:, 0], first_pts[:, 1],
            '-o', color='red',
            markersize=4, linewidth=2
        )

        # 后 12 帧绿色
        second_pts = traj_int[8:8+12]
        ax.plot(
            second_pts[:, 0], second_pts[:, 1],
            '-o', color='green',
            markersize=4, linewidth=2
        )

        # ax.plot(traj_int[:, 0], traj_int[:, 1], '-o', color='red', markersize=4, linewidth=2)
        ax.set_title(f"Global View of Ped: {agent_id} from Frame: {frames[0]}")
        ax.axis('off')
        st.pyplot(fig)
    else:
        st.warning(f"无法读取帧 {frames[0]}，无法可视化轨迹")

    # 4.4 读取所有帧，叠加轨迹，生成 GIF
    st.subheader("GIF 可视化")  # 新增标题
    # 把 GIF 渲染放到一个 placeholder 里
    gif_container = st.empty()

    def make_gif_bytes(images):
        buf = io.BytesIO()
        with imageio.get_writer(buf, format='GIF', mode='I', fps=5) as writer:
            for img in images:
                writer.append_data(img)
        buf.seek(0)
        return buf.read()

    # 准备帧数据（一次即可）
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

    # Replay 按钮——点击后 Streamlit 会重新运行到这里，重新生成并渲染 GIF
    if st.button("Replay"):
        gif_bytes = make_gif_bytes(images_rgb)
        gif_container.image(gif_bytes, caption="当前点动画 (GIF)", use_container_width=True)

    # 4.5 标注输入
    st.markdown("**Prompt 模板示例：**")
    st.code("a person go {forward} then {forward}", language="text")
    st.markdown("**template**")
    # st.code("hesitating", language="text")
    # st.code("forward", language="text")
    # st.code("left", language="text")
    # st.code("right", language="text")
    annotation = st.text_area("请输入本样本的标注内容：", key=f"ann_{scene}_{sample_idx}")
    # 4.6 保存按钮
    if st.button("保存本样本数据", key=f"save_npz_{scene}_{sample_idx}"):
        # 1) 计算 sec_id
        if scene in SDD_FILE:
            sec_id = os.path.basename(os.path.dirname(txt_file))
        else:
            sec_id = scene
        # print(sec_id)
        # 2) 准备要保存的字典
        data_dict = {
            "id":          np.int32(agent_id),
            "start_frame": np.int32(frames[0]),
            "total_seq":   np.int32(len(frames)),
            "traj":        traj.astype(np.float32),
            "start_img":   images_rgb[0].astype(np.uint8),
            "annotation":  np.array(annotation, dtype="U")
        }
        # print(data_dict)
        # 3) 文件名 & 保存
        dataset_name = "SDD" if scene in SDD_FILE else "ETH"
        filename = f"{dataset_name}_{scene}_{sec_id}_{agent_id}.npz"
        os.makedirs("annotations_npz", exist_ok=True)
        save_path = os.path.join("annotations_npz", filename)
        np.savez_compressed(save_path, **data_dict)

        st.success(f"已保存 → {save_path}")