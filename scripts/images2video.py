import sys

import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import mpl_toolkits.mplot3d.art3d as art3d
import matplotlib.animation as animation
from scipy.spatial.transform import Rotation

from dataloader import read_files_replica_onlyImage

def main():
    print("start.")

    rgb_filenames = \
        read_files_replica_onlyImage(folder_path="data/Replica/", data_name="office0_original", num_frames=460, delta=1)

    size = (800, 600)
    name = "outputs/replica_office0.avi"

    out = cv2.VideoWriter(name, cv2.VideoWriter_fourcc(*'XVID'), 30, size)

    for filename in rgb_filenames:
        print(filename)
        img = cv2.imread(filename)
        out.write(img)
    
    out.release()

    print("end.")

if __name__ == "__main__":
    # print("opencv version: " + cv2.__version__)
    main()
