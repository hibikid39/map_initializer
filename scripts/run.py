import sys
import time

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from dataloader import read_files_tum
from initializer.initializer import Initializer
from initializer.bundle_adjustment import BundleAdjustment

class Frame:
    def __init__(self, image, keypoint, description):
        self.image = image 
        self.keypoint = keypoint
        self.description = description
    
    """
    def undistortKeypoints(self, calib, dist):
        new_keypoints
    """

def main():
    print("start.")
    np.random.seed(3407)

    rgb_filenames, camera_params = \
        read_files_tum(folder_path="data/rgbd_dataset_freiburg1_plant/", delta=1)

    orb = cv2.ORB_create(nfeatures=500)

    max_delta = 30
    min_delta = 5

    idx_init = 0
    idx_cur = 0    

    initializer = Initializer(200)
    initialized = False

    frames = []

    num_frames = len(rgb_filenames)
    for i in range(0, num_frames):
        print("frame_idx: ", i)

        image_cur = cv2.imread(rgb_filenames[i], cv2.IMREAD_GRAYSCALE)
        kp, des = orb.detectAndCompute(image_cur, None)
        frame_cur = Frame(image_cur, kp, des)
        frames.append(frame_cur)

        if i < max_delta:
            continue

        #if i < 320:
        #    continue
        
        for j in range(i - max_delta, i - min_delta):
            frame_ref = frames[j]

            kp1 = frame_ref.keypoint
            des1 = frame_ref.description
            kp2 = kp
            des2 = des

            if len(kp2) < 100:
                break
            
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            matches = bf.match(des1, des2)

            if len(matches) < 100:
                break

            initializer.setKeyPoints(kp1, kp2, matches)
            
            if initializer.initialize_F() is True:
                initialized = True

                match_image = cv2.drawMatches(frame_ref.image, kp1, image_cur, kp2, matches[:25], None, flags=2)
                cv2.imwrite("outputs/match_image.jpg", match_image)

                idx_init = j
                idx_cur = i

                break

        if initialized is True:
            break

    if initialized is False:
        exit(1)

    print(rgb_filenames[idx_init], rgb_filenames[idx_cur])
    print(idx_init, idx_cur)
    print(f"R = \n{initializer.R_init}")
    print(f"t = \n{initializer.t_init}")

    # Bundle Adjustment
    ba = BundleAdjustment(pinhole_calib=[initializer.fx, initializer.fy, initializer.cx, initializer.cy])
    rot = Rotation.from_matrix(initializer.R_init)
    camera_param = np.zeros(6)
    camera_param[0:3] = rot.as_rotvec()
    camera_param[3:6] = initializer.t_init
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
