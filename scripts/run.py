import sys
import time

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from dataloader import read_files_tum
from initializer.initializer import Initializer

def main():
    print("start.")
    np.random.seed(3407)

    initializer = Initializer(200)

    rgb_filenames, camera_params = \
        read_files_tum(folder_path="data/rgbd_dataset_freiburg1_desk/", delta=1)

    orb = cv2.ORB_create(nfeatures=500)
    #akaze = cv2.AKAZE_create()

    max_delta = 30  # < 60
    start_frame = 0

    idx_init = 0
    idx_cur = 0

    num_frames = len(rgb_filenames)
    initialized = False

    i = start_frame
    while i < num_frames:
        print("ref frame: ", i)
        image_ref = cv2.imread(rgb_filenames[i], cv2.IMREAD_GRAYSCALE)
        kp1, des1 = orb.detectAndCompute(image_ref, None)
        #kp1, des1 = akaze.detectAndCompute(image_ref, None)
        
        for j in range(i + 3, i + max_delta):
            image_cur = cv2.imread(rgb_filenames[j], cv2.IMREAD_GRAYSCALE)
            kp2, des2 = orb.detectAndCompute(image_cur, None)
            #kp2, des2 = akaze.detectAndCompute(image_cur, None)

            if len(kp2) < 100:
                break
            
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            matches = bf.match(des1, des2)

            if len(matches) < 100:
                break

            initializer.setKeyPoints(kp1, kp2, matches)
            
            if initializer.initialize() is True:
                initialized = True

                match_image = cv2.drawMatches(image_ref, kp1, image_cur, kp2, matches[:25], None, flags=2)
                cv2.imwrite("outputs/match_image.jpg", match_image)

                idx_init = i
                idx_cur = j

                break

        if initialized is True:
            break
        else:
            i += max_delta

    print(rgb_filenames[idx_init], rgb_filenames[idx_cur])
    print(idx_init, idx_cur)
    print(f"R = \n{initializer.R_init}")
    print(f"t = \n{initializer.t_init}")

    rotX_90 = Rotation.from_rotvec(np.pi/2 * np.array([1, 0, 0]))

    rot_vec_cur = camera_params[idx_cur, 0:3]
    translation_cur= camera_params[idx_cur, 3:6]
    transform_cur= np.identity(4)
    rot_cur = Rotation.from_rotvec(rot_vec_cur)
    rot_cur = rotX_90 * rot_cur  # from TUM coord to OpenCV coord
    transform_cur[:3, :3] = rot_cur.as_matrix()
    transform_cur[:3, 3] = translation_cur

    rot_vec_init = camera_params[idx_init, 0:3]
    translation_init = camera_params[idx_init, 3:6]
    transform_init = np.identity(4)
    rot_init = Rotation.from_rotvec(rot_vec_init)
    rot_init = rotX_90 * rot_init  # from TUM coord to OpenCV coord
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
