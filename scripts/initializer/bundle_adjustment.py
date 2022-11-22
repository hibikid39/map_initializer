import numpy as np
import math
from scipy.sparse import lil_matrix
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from util import *

# Compute residuals
def func(params, n_cameras, n_points, camera_indices, point_indices, points_2d, pinhole_calib):
    camera_params = params[:n_cameras * 6].reshape((n_cameras, 6))
    points_3d = params[n_cameras * 6:].reshape((n_points, 3))
    
    N = len(point_indices)
    residuals = np.zeros((N, 2))
    for i in range(0, N):
        point_inx = point_indices[i]
        camera_idx = camera_indices[i]
        point_proj = BundleAdjustment.project(points_3d[point_inx], camera_params[camera_idx], pinhole_calib)
        residuals[i] = point_proj - points_2d[i]

    return residuals.ravel()

class BundleAdjustment:
    def __init__(self, pinhole_calib):
        self.pinhole_calib = pinhole_calib
        self.n_cameras = 2

    def setParams(self, keyPoints1_cv, keyPoints2_cv, matches_cv, points3d, bGoods, camera_param):
        self.camera_params = np.zeros((2, 6))
        self.camera_params[1] = camera_param
        self.points_3d = []

        self.camera_indices = []
        self.point_indices = []
        self.points_2d = []
        self.n_points = 0

        for match in matches_cv:
            if bGoods[match.queryIdx] is True:
                self.points_3d.append(points3d[match.queryIdx])
                self.n_points += 1

                # camera 1 (init)
                self.camera_indices.append(0)
                self.point_indices.append(len(self.points_3d) - 1)
                self.points_2d.append(keyPoints1_cv[match.queryIdx].pt)

                # camera 2 (cur)
                self.camera_indices.append(1)
                self.point_indices.append(len(self.points_3d) - 1)
                self.points_2d.append(keyPoints2_cv[match.trainIdx].pt)

        self.camera_indices = np.array(self.camera_indices)
        self.point_indices = np.array(self.point_indices)
        self.points_2d = np.array(self.points_2d)

        self.points_3d = np.array(self.points_3d)
        self.params = np.zeros(self.camera_params.size + self.points_3d.size)
        self.params[:self.camera_params.size] = self.camera_params.flatten()
        self.params[self.camera_params.size:] = self.points_3d.flatten()

    # Convert 3D points to 2D by projecting onto images
    @classmethod
    def project(cls, point3d_w, camera_param, pinhole_calib):
        R, t = Twist2RT(camera_param)
        point3d_c = R @ point3d_w + t # camera coordinate

        fx = pinhole_calib[0]
        fy = pinhole_calib[1]
        cx = pinhole_calib[2]
        cy = pinhole_calib[3]

        invZ = 1.0 / point3d_c[2]
        image_x = fx * point3d_c[0] * invZ + cx # image coordinate
        image_y = fy * point3d_c[1] * invZ + cy

        return np.array([image_x, image_y])

    def bundle_adjustment_sparsity(self):
        m = self.camera_indices.size * 2
        n = self.n_cameras * 6 + self.n_points * 3
        A = lil_matrix((m, n), dtype=int)

        #i = np.arange(self.camera_indices.size)
        for s in range(6):
            for i in range(self.camera_indices.size):
                if self.camera_indices[i] == 1:
                    A[2 * i, self.camera_indices[i] * 6 + s] = 1
                    A[2 * i + 1, self.camera_indices[i] * 6 + s] = 1

        for s in range(3):
            A[2 * i, self.n_cameras * 6 + self.point_indices * 3 + s] = 1
            A[2 * i + 1, self.n_cameras * 6 + self.point_indices * 3 + s] = 1

        return A

    def optimize(self):
        print("start Full BA")
        A = self.bundle_adjustment_sparsity()

        res = least_squares(func, self.params, jac_sparsity=A, verbose=0, x_scale='jac', ftol=1e-8, method='trf',
                    args=(self.n_cameras, self.n_points, self.camera_indices, self.point_indices, self.points_2d, self.pinhole_calib))

        res_param = res.x
        camera_params = res_param[:self.n_cameras * 6].reshape((self.n_cameras, 6))
        points_3d = res_param[self.n_cameras * 6:].reshape((self.n_points, 3))

        R, t = Twist2RT(camera_params[1])
        """
        rot = Rotation.from_rotvec(camera_params[1, 0:3])
        rot_mat = rot.as_matrix()
        translation = camera_params[1, 3:6]
        """

        print(R)
        print(t)
