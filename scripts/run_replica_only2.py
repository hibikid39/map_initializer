import sys
import time

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from dataloader import read_files_replica
from initializer.initializer import Initializer
from initializer.bundle_adjustment import BundleAdjustment
from util import *
from frame import Frame

def main():
    print("start.")
    np.random.seed(3407)

    rgb_filenames, camera_params, camera_calib = \
        read_files_replica(folder_path="data/Replica/", data_name="office0_original", delta=15)
    camera_calib = camera_calib[2:6]

    orb = cv2.ORB_create(nfeatures=500)

    initializer = Initializer(200)
    initializer.setK(camera_calib)
    
    image_cur = cv2.imread(rgb_filenames[1], cv2.IMREAD_GRAYSCALE)
    kp_cur, des_cur = orb.detectAndCompute(image_cur, None)
    frame_cur = Frame(image_cur, kp_cur, des_cur)

    image_old = cv2.imread(rgb_filenames[0], cv2.IMREAD_GRAYSCALE)
    kp_old, des_old = orb.detectAndCompute(image_old, None)
    frame_old = Frame(image_old, kp_old, des_old)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des_old, des_old)

    initializer.setKeyPoints(kp_old, kp_old, matches)
            
    initializer.initialize_F()

    match_image = cv2.drawMatches(frame_old.image, kp_old, frame_cur.image, kp_cur, matches[:25], None, flags=2)
    cv2.imwrite("outputs/match_image.jpg", match_image)

    return

    print(rgb_filenames[idx_init], rgb_filenames[idx_cur])
    print(idx_init, idx_cur)
    print(f"R = \n{initializer.R_init}")
    print(f"t = \n{initializer.t_init}")

    # Bundle Adjustment
    ba = BundleAdjustment(pinhole_calib=[initializer.fx, initializer.fy, initializer.cx, initializer.cy])
    camera_param = RT2Twist(initializer.R_init, initializer.t_init)
    ba.setParams(initializer.keyPoints1_cv, initializer.keyPoints2_cv, initializer.matches_cv, initializer.points3d, initializer.bGoods, camera_param)
    ba.optimize()

    rot_vec_cur = camera_params[idx_cur, 0:3]
    translation_cur= camera_params[idx_cur, 3:6]
    transform_cur= np.identity(4)
    rot_cur = Rotation.from_rotvec(rot_vec_cur)
    transform_cur[:3, :3] = rot_cur.as_matrix()
    transform_cur[:3, 3] = translation_cur

    rot_vec_init = camera_params[idx_init, 0:3]
    translation_init = camera_params[idx_init, 3:6]
    transform_init = np.identity(4)
    rot_init = Rotation.from_rotvec(rot_vec_init)
    transform_init[:3, :3] = rot_init.as_matrix()
    transform_init[:3, 3] = translation_init

    transform_rel = np.linalg.inv(transform_cur) @ transform_init
    transform_rel[:3, 3] = transform_rel[:3, 3] / np.linalg.norm(transform_rel[:3, 3])

    print("transform_rel = ")
    print(transform_rel)

    print("end.")

if __name__ == "__main__":
    # print("opencv version: " + cv2.__version__)
    # np.show_config()
    main()
