import sys

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

    orb = cv2.ORB_create()

    max_delta = 10  # < 60
    start_frame = 90

    idx_init = 0
    idx_cur = 0

    num_frames = len(rgb_filenames)
    initialized = False

    for i in range(start_frame, num_frames - 60):
        print("ref frame: ", i)
        image_ref = cv2.imread(rgb_filenames[i], cv2.IMREAD_GRAYSCALE)
        kp1, des1 = orb.detectAndCompute(image_ref, None) # max num of feataure points: 500(default)
        
        for j in range(i + 5, i + max_delta):
            image_cur = cv2.imread(rgb_filenames[j], cv2.IMREAD_GRAYSCALE)
            kp2, des2 = orb.detectAndCompute(image_cur, None)

            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            matches = bf.match(des1, des2)
            matches = sorted(matches, key = lambda x:x.distance)

            """
            # FLANN match
            FLANN_INDEX_LSH = 6
            index_params= dict(algorithm = FLANN_INDEX_LSH,
                                table_number = 6, # 12
                                key_size = 14,     # 20
                                multi_probe_level = 1) #2
            search_params = dict(checks=50)
            flann = cv2.FlannBasedMatcher(index_params,search_params)
            knn_matches = flann.knnMatch(des1, des2, k=2)
            # ratio test as per Lowe's paper
            matches = []
            for i in range(0, len(knn_matches)):
                try:
                    m, n = knn_matches[i]
                    if m.distance < 0.7 * n.distance:
                        matches.append(m)
                except:
                    continue
            """

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

    print(initialized)
    print(rgb_filenames[idx_init], rgb_filenames[idx_cur])
    print(idx_init, idx_cur)
    print(f"R = \n{initializer.R_init}")
    print(f"t = \n{initializer.t_init}")

    rot_vec_cur = camera_params[idx_cur, 0:3]
    translation_cur= camera_params[idx_cur, 3:6]
    transform_cur= np.identity(4)
    transform_cur[:3, :3] = Rotation.from_rotvec(rot_vec_cur).as_matrix()
    transform_cur[:3, 3] = translation_cur

    rot_vec_init = camera_params[idx_init, 0:3]
    translation_init = camera_params[idx_init, 3:6]
    transform_init = np.identity(4)
    transform_init[:3, :3] = Rotation.from_rotvec(rot_vec_init).as_matrix()
    transform_init[:3, 3] = translation_init

    transform_rel = transform_cur @ np.linalg.inv(transform_init)
    transform_rel[:3, 3] = transform_rel[:3, 3] / np.linalg.norm(transform_rel[:3, 3])

    """
    rot_vec_cur = camera_params[idx_cur, 0:3]
    trans_cur = camera_params[idx_cur, 3:6]

    rot_init = Rotation.from_rotvec(rot_vec_init)
    rot_cur = Rotation.from_rotvec(rot_vec_cur)
    rot_rel = rot_init.inv() * rot_cur

    trans_rel = trans_cur - trans_init
    trans_rel = trans_rel / np.linalg.norm(trans_rel)
    trans_rel = rot_init.as_matrix() @ trans_rel
    """

    print("transform_rel = ")
    print(transform_rel)

    print("end.")

if __name__ == "__main__":
    # print("opencv version: " + cv2.__version__)
    main()
