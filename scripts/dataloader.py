import csv

import cv2
import numpy as np
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation
import json

# load data files in TUM directories
def read_files_tum(folder_path: str = "data/rgbd_dataset_freiburg1_desk/", delta: int = 1):
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
