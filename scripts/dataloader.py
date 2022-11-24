import csv

import cv2
import numpy as np
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation
import json
from util import *

# load data files in TUM directories
def read_files_tum(folder_path: str = "data/TUM/rgbd_dataset_freiburg1_desk/", delta: int = 1):

    # rgb file pathes
    csv_file = open(folder_path + "rgb.txt", "r")
    f = csv.reader(csv_file, delimiter=" ")
    next(f)  # skip headers
    next(f)
    next(f)
    rgb_filenames = []
    rgb_timestamps_list = []
    for row in f:
        rgb_filenames.append("{}{}".format(folder_path, row[1]))
        rgb_timestamps_list.append(row[0])
    rgb_timestamps = np.array(rgb_timestamps_list)

    # pose file pathes
    csv_file = open(folder_path + "groundtruth.txt", "r")
    f = csv.reader(csv_file, delimiter=" ")
    next(f)  # skip headers
    next(f)
    next(f)
    gt_list = []
    for row in f:
        gt_list.append(np.array(row).astype(float))
    gt = np.array(gt_list)  # [:, [0, 2,3,1,5,6,4,7]]

    # interpolate gt pose
    interp_pose = interp1d(gt[:, 0], gt[:, 1:].transpose(1, 0))
    poses = interp_pose(rgb_timestamps).transpose(1, 0)

    # set camera_param
    camera_params = np.zeros((poses.shape[0], 6), np.float32) # [[rot vec], [rot trans]]
    rots = Rotation.from_quat(poses[:, 3:])
    camera_params[:, :3] = rots.as_rotvec()
    camera_params[:, 3:6] = poses[:, :3]

    return rgb_filenames[::delta], camera_params[::delta]

# load data files in TUM directories
def read_files_replica(folder_path: str = "data/Replica/", data_name: str = "room1", delta: int = 1):
    # load intrinsic params
    camera_calib = np.zeros(7)
    with open(folder_path + "cam_params.json") as f:
        df = json.load(f)
        camera_calib[0] = df["camera"]["w"]
        camera_calib[1] = df["camera"]["h"]
        camera_calib[2] = df["camera"]["fx"]
        camera_calib[3] = df["camera"]["fy"]
        camera_calib[4] = df["camera"]["cx"]
        camera_calib[5] = df["camera"]["cy"]
        camera_calib[6] = df["camera"]["scale"]
        
    # load trajectories
    csv_file = open(folder_path + data_name + "/traj.txt", "r")
    f = csv.reader(csv_file, delimiter=" ")
    trajectories_list = []
    for row in f:
        trajectories_list.append(np.array(row).astype(float))
    trajectories = np.array(trajectories_list)
    trajectories = trajectories.reshape((-1, 4, 4))
    camera_params = np.zeros((trajectories.shape[0], 6))
    for i, pose in enumerate(trajectories):
        rot = Rotation.from_matrix(pose[0:3, 0:3])
        rot_vec = rot.as_rotvec()
        camera_params[i, 0:3] = rot_vec
        camera_params[i, 3:6] = pose[:3, 3]

    # set rgb filenames
    num_frames = 2000
    rgb_filenames = []
    for i in range(0, num_frames):
        filename = folder_path + data_name + "/results/frame{:0>6}.jpg".format(i)
        rgb_filenames.append(filename)
    
    return rgb_filenames[::delta], camera_params[::delta], camera_calib
    