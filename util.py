import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch

def load_sdd_paths(sdd_root, scene_list):
    """
    遍历 SDD 数据集目录，返回两个 dict：
      - scenario_txt_path[scene]   = [所有 segment 下的 ann.txt 路径]
      - scenario_video_path[scene] = [所有 segment 下的 video.mov 路径]
    """
    scenario_txt_path = {}
    scenario_video_path = {}

    for scene in scene_list:
        ann_scene_dir = os.path.join(sdd_root, 'annotations', scene)
        vid_scene_dir = os.path.join(sdd_root, 'videos', scene)
        txt_paths = []
        video_paths = []

        if not os.path.isdir(ann_scene_dir) or not os.path.isdir(vid_scene_dir):
            # 如果某个子目录不存在，就跳过
            continue

        # annotations/scene 下，每个子文件夹都是一个 segment
        for segment in sorted(os.listdir(ann_scene_dir)):
            ann_seg_dir = os.path.join(ann_scene_dir, segment)
            vid_seg_dir = os.path.join(vid_scene_dir, segment)

            txt_file = os.path.join(ann_seg_dir, 'annotations.txt')
            vid_file = os.path.join(vid_seg_dir, 'video.mov')
            # 确保两个文件都存在才加入
            if os.path.isfile(txt_file) and os.path.isfile(vid_file):
                txt_paths.append(txt_file)
                video_paths.append(vid_file)
        
        scenario_txt_path[scene] = txt_paths
        scenario_video_path[scene] = video_paths
    return scenario_txt_path, scenario_video_path

def load_eth_paths(eth_root, scene_list=None):
    scenario_txt_path = {}
    scenario_video_path = {}

    # 如果未指定 scene_list，则默认加载根目录下所有子文件夹
    if scene_list is None:
        scene_list = [d for d in sorted(os.listdir(eth_root))
                      if os.path.isdir(os.path.join(eth_root, d))]

    for scene in scene_list:
        scene_dir = os.path.join(eth_root, scene)
        if not os.path.isdir(scene_dir):
            # 不存在则跳过
            continue

        txt_paths = []
        video_paths = []

        # 遍历场景文件夹下的所有文件
        for fname in sorted(os.listdir(scene_dir)):
            fpath = os.path.join(scene_dir, fname)
            if not os.path.isfile(fpath):
                continue

            lower = fname.lower()
            if lower.endswith('.txt'):
                txt_paths.append(fpath)
            elif lower.endswith('.mov') or lower.endswith('.avi'):
                video_paths.append(fpath)

        scenario_txt_path[scene]   = txt_paths
        scenario_video_path[scene] = video_paths

    return scenario_txt_path, scenario_video_path


def visualize_trajectory_on_frame(frame: np.ndarray,
                                  trajectory: np.ndarray,
                                  save_path: str):
    """
    在给定图像上画出轨迹并保存。
    - frame: HxWx3 的 RGB 图像
    - trajectory: N×2 的像素坐标数组 (x, y)
    - save_path: 输出文件路径 (.png)
    """
    plt.figure(figsize=(6,6))
    plt.imshow(frame)
    traj = trajectory.astype(int)
    # 画线和点
    plt.plot(traj[:,0], traj[:,1], '-o', markersize=4, linewidth=2)
    plt.axis('off')
    plt.tight_layout(pad=0)
    plt.savefig(save_path, dpi=150)
    plt.close()

def generate_H_data_(dataset_name):
    H_data = []
    if dataset_name == 'STUDENT':
        H_data = np.array([[0.02220407,0,-0.48],
                    [0,-0.02477289,13.92551292],
                    [0,0,1]]) #students(001/003)
    elif dataset_name == 'ZARA01':
        H_data = np.array([[0.02174104 ,0,-0.15],
                        [0,-0.02461883,13.77429807],
                        [0,0,1]]) #crowds_zara01
    elif dataset_name == 'ZARA02':
        H_data = np.array([[0.02174104,0,-0.4],
                        [0,-0.02386598,14.98401686],
                        [0,0,1]]) #crowds_zara02/03
    elif dataset_name == 'ETH':
        H_data = np.array([[2.8128700e-02,2.0091900e-03,-4.6693600e+00],
                        [8.0625700e-04,2.5195500e-02,-5.0608800e+00],
                        [3.4555400e-04,9.2512200e-05,4.6255300e-01]]) #biwi_eth
    elif dataset_name == 'HOTEL':
        H_data = np.array([[1.1048200e-02,6.6958900e-04,-3.3295300e+00],
                        [-1.5966000e-03,1.1632400e-02,-5.3951400e+00],
                        [1.1190700e-04,1.3617400e-05,5.4276600e-01]]) #biwi_hotel
    else:
        print('check H_data infor!')
    # print('current H data: ', dataset_name)
    return H_data

def generate_dataset_tag(datasets_path):
        normalized_path = os.path.normpath(datasets_path)

        # 提取文件名（包含扩展名）
        filename = os.path.basename(normalized_path)

        # 分离文件名和扩展名
        dataset_name, extension = os.path.splitext(filename)

        h_data = generate_H_data_(dataset_name)
        if dataset_name == 'ETH' or dataset_name == 'HOTEL':
            xy_tag = 0
        else:
            xy_tag = 1
        return h_data, xy_tag

def trajectory2pixel(traj_data, H_data):
    # traj_data input size [x,2]
    # H_data input size [3,3]
    trajectory_data = traj_data
    N = len(trajectory_data)
    # 创建形状为 (N, 1) 的全 1 列向量
    ones_column = np.ones((N, 1))
    # 拼接数据，结果形状为 (N, 3)
    data = np.hstack((trajectory_data, ones_column))
    inv_H_data = np.linalg.inv(H_data).T
    pixel_traj = np.dot(data, inv_H_data)

    epsilon = 1e-10  # To prevent division by zero
    denom = pixel_traj[:, 2][:, np.newaxis] + epsilon

    pixel_traj_normalized = pixel_traj[:, :2] / denom
    # return the pixeled pos from traj
    return pixel_traj_normalized.astype(int)