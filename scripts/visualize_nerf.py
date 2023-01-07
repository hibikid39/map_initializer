import sys

import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import mpl_toolkits.mplot3d.art3d as art3d
import matplotlib.animation as animation
from scipy.spatial.transform import Rotation

from dataloader import read_files_nerf

class Arrow:
    def __init__(self, length = 1.0):
        self.length = length
        self.root = np.array([0.0, 0.0, 0.0])
        self.pointedX = self.root + np.array([length, 0.0, 0.0])
        self.pointedY = self.root + np.array([0.0, length, 0.0])
        self.pointedZ = self.root + np.array([0.0, 0.0, length])
    
    def transform(self, R, t):
        self.root = R @ self.root
        self.pointedX = R @ self.pointedX
        self.pointedY = R @ self.pointedY
        self.pointedZ = R @ self.pointedZ

        self.root = self.root + t
        self.pointedX = self.pointedX + t
        self.pointedY = self.pointedY + t
        self.pointedZ = self.pointedZ + t

    def draw_coord(self, ax):
        xline = art3d.Line3D([self.root[0], self.pointedX[0]],[self.root[1], self.pointedX[1]],[self.root[2], self.pointedX[2]], color='red')
        yline = art3d.Line3D([self.root[0], self.pointedY[0]],[self.root[1], self.pointedY[1]],[self.root[2], self.pointedY[2]], color='green')
        zline = art3d.Line3D([self.root[0], self.pointedZ[0]],[self.root[1], self.pointedZ[1]],[self.root[2], self.pointedZ[2]], color='blue')

        ax.add_line(xline)
        ax.add_line(yline)
        ax.add_line(zline)

def main():
    print("start.")

    rgb_filenames, camera_params = \
        read_files_nerf(folder_path="data/nerf_synthetic/lego/", delta=1)

    scale = 1
    camera_params[:, 3:6] *= scale

    ax = plt.figure().add_subplot(projection='3d')

    ax.plot(camera_params[:, 3], camera_params[:, 4], camera_params[:, 5], "o", color="#A0A0A0", ms=1)

    for param in camera_params:
        rot = Rotation.from_rotvec(param[0:3])
        R = rot.as_matrix()
        #R = np.linalg.inv(R)
        t = param[3:6]

        arrow = Arrow(0.5)
        arrow.transform(R, t)
        arrow.draw_coord(ax)

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    ax.view_init(elev=90, azim=0)

    plt.savefig("outputs/nerf1.png", format="png", dpi=300)

    print("end.")

if __name__ == "__main__":
    # print("opencv version: " + cv2.__version__)
    main()
