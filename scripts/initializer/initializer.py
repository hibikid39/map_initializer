import time

import cv2
import numpy as np
from numpy.linalg import svd
import math

class Initializer:
    def __init__(self, maxIteration) -> None:
        self.maxIteration = maxIteration  # num of iteration for RANSAC scheme
        self.sigma = 1.0  # TODO: what is this
        self.R_init = np.zeros((3, 3))
        self.t_init = np.zeros(3)
        
        self.setK()

    def setKeyPoints(self, keyPoints1_cv, keyPoints2_cv, matches_cv):
        self.keyPoints1_cv = keyPoints1_cv
        self.keyPoints2_cv = keyPoints2_cv
        self.matches_cv = matches_cv

    def setK(self):
        # for TUM RGBD dataset
        self.K = np.identity(3)
        self.K[0, 0] = self.fx = 517.3  # = 525.0
        self.K[1, 1] = self.fy = 516.5  # = 525.0
        self.K[0, 2] = self.cx = 318.6  # = 319.5
        self.K[1, 2] = self.cy = 255.3  # = 239.5

    def initialize_F(self):
        # generate sets of 8 points for each RANSAC iteration
        sets_idx = np.random.randint(0, len(self.matches_cv)-1, (self.maxIteration, 8))

        # compute a fundamental matrix
        scoreF, bInliersF, F = self.findFundamental(sets_idx)

        minParallax = 1.0
        minTriangulated = 50.0

        return self.reconstructF(bInliersF, F, minParallax, minTriangulated)

    def initialize(self):
        # generate sets of 8 points for each RANSAC iteration
        sets_idx = np.random.randint(0, len(self.matches_cv)-1, (self.maxIteration, 8))

        # compute a fundamental matrix and a homography
        scoreH, bInliersH, H = self.findHomography(sets_idx)
        scoreF, bInliersF, F = self.findFundamental(sets_idx)

        # compute ratio of score
        R_H = scoreH / (scoreH + scoreF)
        # print("R_H = ", R_H)

        minParallax = 1.0
        minTriangulated = 50.0

        # try to reconstruct transfomation from F and H
        if R_H > 0.45:
            return self.reconstructH(bInliersH, H, minParallax, minTriangulated)
        else:
            return self.reconstructF(bInliersF, F, minParallax, minTriangulated)


    def findHomography(self, sets_idx):
        # Normalize coordinates
        normalizedPoints1_cv, T1 = self.normalize(self.keyPoints1_cv)
        normalizedPoints2_cv, T2 = self.normalize(self.keyPoints2_cv)
        T2inv = np.linalg.inv(T2)

        # best score
        bestScore = 0

        for i in range(0, self.maxIteration):
            set_points1_cv = []
            set_points2_cv = []

            for j in range(0, 8):
                idx = sets_idx[i, j]
                set_points1_cv.append(normalizedPoints1_cv[self.matches_cv[idx].queryIdx])
                set_points2_cv.append(normalizedPoints2_cv[self.matches_cv[idx].trainIdx])
            
            Hn = self.computeH(set_points1_cv, set_points2_cv)
            H21i = T2inv @ Hn @ T1
            H12i = np.linalg.inv(H21i)
            """
            try:
                H12i = np.linalg.inv(H21i)
            except Exception as e:
                print(e)
                H21i += np.identity(3) * 0.0001
                H12i = np.linalg.inv(H21i)
            """ 
            score, bInliers = self.checkHomography(H21i, H12i)

            if score > bestScore:
                H21 = H21i.copy()
                best_bInliers = bInliers
                bestScore = score

        return bestScore, best_bInliers, H21
    
    def findFundamental(self, sets_idx):
        # Normalize coordinates
        normalizedPoints1_cv, T1 = self.normalize(self.keyPoints1_cv)
        normalizedPoints2_cv, T2 = self.normalize(self.keyPoints2_cv)
        T2t = T2.transpose()

        # best score
        bestScore = 0

        for i in range(0, self.maxIteration):
            set_points1_cv = []
            set_points2_cv = []

            for j in range(0, 8):
                idx = sets_idx[i, j]
                set_points1_cv.append(normalizedPoints1_cv[self.matches_cv[idx].queryIdx])
                set_points2_cv.append(normalizedPoints2_cv[self.matches_cv[idx].trainIdx])
            
            Fn = self.computeF(set_points1_cv, set_points2_cv)
            F21i = T2t @ Fn @ T1
            
            score, bInliers = self.checkFundamental(F21i)

            if score > bestScore:
                F21 = F21i.copy()
                best_bInliers = bInliers
                bestScore = score
        
        return bestScore, best_bInliers, F21

    def computeH(self, points1_cv, points2_cv):
        N = len(points1_cv)  # N=8
        A = np.zeros((N*2, 9))

        for i in range(0, N):
            u1 = points1_cv[i][0]
            v1 = points1_cv[i][1]
            u2 = points2_cv[i][0]
            v2 = points2_cv[i][1]

            A[2*i, 0] = 0.0
            A[2*i, 1] = 0.0
            A[2*i, 2] = 0.0
            A[2*i, 3] = -u1
            A[2*i, 4] = -v1
            A[2*i, 5] = -1.0
            A[2*i, 6] = v2*u1
            A[2*i, 7] = v2*v1
            A[2*i, 8] = v2

            A[2*i+1, 0] = u1
            A[2*i+1, 1] = v1
            A[2*i+1, 2] = 1.0
            A[2*i+1, 3] = 0.0
            A[2*i+1, 4] = 0.0
            A[2*i+1, 5] = 0.0
            A[2*i+1, 6] = -u2*u1
            A[2*i+1, 7] = -u2*v1
            A[2*i+1, 8] = -u2

        u, s, vt = svd(A)
        #print(f"s={s}")
        #print(f"vt={vt}, vt.shape={vt.shape}")

        return vt[8].reshape((3, 3))
        

    def computeF(self, points1_cv, points2_cv):
        N = len(points1_cv)  # N=8
        A = np.zeros((N, 9))

        for i in range(0, N):
            u1 = points1_cv[i][0]
            v1 = points1_cv[i][1]
            u2 = points2_cv[i][0]
            v2 = points2_cv[i][1]

            A[i, 0] = u2*u1
            A[i, 1] = u2*v1
            A[i, 2] = u2
            A[i, 3] = v2*u1
            A[i, 4] = v2*v1
            A[i, 5] = v2
            A[i, 6] = u1
            A[i, 7] = v1
            A[i, 8] = 1

        u, s, vt = svd(A)
        Fpre = vt[8].reshape((3, 3))
        u, s, vt = svd(Fpre)
        s[2] = 0

        return u @ np.diag(s) @ vt
    
    def checkHomography(self, H21, H12):
        h11 = H21[0, 0]
        h12 = H21[0, 1]
        h13 = H21[0, 2]
        h21 = H21[1, 0]
        h22 = H21[1, 1]
        h23 = H21[1, 2]
        h31 = H21[2, 0]
        h32 = H21[2, 1]
        h33 = H21[2, 2]

        h11inv = H21[0, 0]
        h12inv = H12[0, 1]
        h13inv = H12[0, 2]
        h21inv = H12[1, 0]
        h22inv = H12[1, 1]
        h23inv = H12[1, 2]
        h31inv = H12[2, 0]
        h32inv = H12[2, 1]
        h33inv = H12[2, 2]

        N = len(self.matches_cv)
        bInliers = [False for _ in range(0, N)]

        score = 0
        th = 5.991
        invSigmaSquare = 1.0 / (self.sigma * self.sigma)

        for i in range(0 ,N):
            bIn = True

            kp1 = self.keyPoints1_cv[self.matches_cv[i].queryIdx]
            kp2 = self.keyPoints2_cv[self.matches_cv[i].trainIdx]

            u1 = kp1.pt[0]
            v1 = kp1.pt[1]
            u2 = kp2.pt[0]
            v2 = kp2.pt[1]

            # Reprojection error in first image
            w2in1inv = 1.0 / (h31inv * u2 + h32inv * v2 + h33inv)
            u2in1 = (h11inv * u2 + h12inv * v2 + h13inv) * w2in1inv
            v2in1 = (h21inv * u2 + h22inv * v2 + h23inv) * w2in1inv
            squareDist1 = (u1 - u2in1) * (u1 - u2in1) + (v1 - v2in1) * (v1 - v2in1)
            chiSquare1 = squareDist1 * invSigmaSquare
            if chiSquare1 > th:
                bIn = False
            else:
                score += th - chiSquare1

            # Reprojection error in second image
            # x1in2 = H21*x1
            w1in2inv = 1.0 / (h31 * u1 + h32 * v1 + h33)
            u1in2 = (h11 * u1 + h12 * v1 + h13) * w1in2inv
            v1in2 = (h21 * u1 + h22 * v1 + h23) * w1in2inv
            squareDist2 = (u2 - u1in2) * (u2 - u1in2) + (v2 - v1in2) * (v2 - v1in2)
            chiSquare2 = squareDist2 * invSigmaSquare
            if chiSquare2 > th:
                bIn = False
            else:
                score += th - chiSquare2

            if bIn is True:
                bInliers[i] = True
            else:
                bInliers[i] = False

        return score, bInliers


    def checkFundamental(self, F21):
        f11 = F21[0, 0]
        f12 = F21[0, 1]
        f13 = F21[0, 2]
        f21 = F21[1, 0]
        f22 = F21[1, 1]
        f23 = F21[1, 2]
        f31 = F21[2, 0]
        f32 = F21[2, 1]
        f33 = F21[2, 2]

        N = len(self.matches_cv)
        bInliers = [False for _ in range(0, N)]

        score = 0
        th = 3.841
        thScore = 5.991
        invSigmaSquare = 1.0 / (self.sigma * self.sigma)

        for i in range(0, N):
            bIn = True
            kp1 = self.keyPoints1_cv[self.matches_cv[i].queryIdx]
            kp2 = self.keyPoints2_cv[self.matches_cv[i].trainIdx]

            u1 = kp1.pt[0]
            v1 = kp1.pt[1]
            u2 = kp2.pt[0]
            v2 = kp2.pt[1]

            # Reprojection error in second image
            # l2=F21x1=(a2,b2,c2)
            a2 = f11 * u1 + f12 * v1 + f13
            b2 = f21 * u1 + f22 * v1 + f23
            c2 = f31 * u1 + f32 * v1 + f33
            num2 = a2 * u2 + b2 * v2 + c2
            squareDist1 = num2 * num2 / (a2 * a2 + b2 * b2)
            chiSquare1 = squareDist1 * invSigmaSquare
            if chiSquare1 > th:
                bIn = False
            else:
                score += thScore - chiSquare1

            # Reprojection error in second image
            # l1 =x2tF21=(a1,b1,c1)
            a1 = f11 * u2 + f21 * v2 + f31
            b1 = f12 * u2 + f22 * v2 + f32
            c1 = f13 * u2 + f23 * v2 + f33
            num1 = a1 * u1 + b1 * v1 + c1
            squareDist2 = num1 * num1 / (a1 * a1 + b1 * b1)
            chiSquare2 = squareDist2 * invSigmaSquare
            if chiSquare2 > th:
                bIn = False
            else:
                score += thScore - chiSquare2

            if bIn is True:
                bInliers[i] = True
            else:
                bInliers[i] = False

        return score, bInliers

    def reconstructH(self, bInliers, H, minParallax = 1.0, minTriangulated = 50.0):
        print("[reconstructH]")
        return False
    
    def reconstructF(self, bInliers, F, minParallax = 1.0, minTriangulated = 50.0):
        N = bInliers.count(True)  # num of inliers

        # compute essential matrix
        E = self.K.transpose() @ F @ self.K

        # recover the 4 motion hypotheses
        t1, t2, R1, R2 = self.decomposeE(E)

        # check rotation and translation
        nGood1, parallax1, points3d1, bGoods1 = self.checkRT(R1, t1, bInliers)
        nGood2, parallax2, points3d2, bGoods2 = self.checkRT(R2, t1, bInliers)
        nGood3, parallax3, points3d3, bGoods3 = self.checkRT(R1, t2, bInliers)
        nGood4, parallax4, points3d4, bGoods4 = self.checkRT(R2, t2, bInliers)

        maxGood = max(nGood1, nGood2, nGood3, nGood4)

        nMinGood = max(0.9 * N, minTriangulated)

        # to get clear winner
        nsimilar = 0
        if nGood1 > 0.7 * maxGood:
            nsimilar += 1
        if nGood2 > 0.7 * maxGood:
            nsimilar += 1
        if nGood3 > 0.7 * maxGood:
            nsimilar += 1
        if nGood4 > 0.7 * maxGood:
            nsimilar += 1

        # If there is not a clear winner or not enough triangulated points reject initialization
        if(nsimilar > 1 or maxGood < nMinGood):
            return False

        # If best reconstruction has enough parallax initialize
        if maxGood == nGood1:
            if parallax1 > minParallax:
                self.R_init = R1
                self.t_init = t1
                return True
        elif maxGood == nGood2:
            if parallax2 > minParallax:
                self.R_init = R2
                self.t_init = t1
                return True
        elif maxGood == nGood3:
            if parallax3 > minParallax:
                self.R_init = R1
                self.t_init = t2
                return True
        elif maxGood == nGood4:
            if parallax4 > minParallax:
                self.R_init = R2
                self.t_init = t2
                return True

        return False
    
    # calculate 3d point
    def triangulate(self, kp1, kp2, P1, P2):
        A = np.zeros((4, 4))

        A[0, :] = kp1.pt[0] * P1[2, :] - P1[0, :]
        A[1, :] = kp1.pt[1] * P1[2, :] - P1[1, :]
        A[2, :] = kp2.pt[0] * P2[2, :] - P2[0, :]
        A[3, :] = kp2.pt[1] * P2[2, :] - P2[1, :]

        u, s, vt = svd(A)
        point3d = vt[3, :].transpose()
        point3d = point3d[:3] / point3d[3]
        return point3d

    def normalize(self, keyPoints_cv):
        meanX = 0
        meanY = 0
        N = len(keyPoints_cv)

        normalizedPoints = []

        for i in range(0, N):
            meanX += keyPoints_cv[i].pt[0]
            meanY += keyPoints_cv[i].pt[1]

        meanX = meanX / N
        meanY = meanY / N

        meanDevX = 0
        meanDevY = 0

        for i in range(0, N):
            normalizedPoints.append([keyPoints_cv[i].pt[0] - meanX,
                                     keyPoints_cv[i].pt[1] - meanY])
            meanDevX += abs(normalizedPoints[i][0])
            meanDevY += abs(normalizedPoints[i][1])

        meanDevX = meanDevX / N
        meanDevY = meanDevY / N

        sX = 1.0 / meanDevX
        sY = 1.0 / meanDevY

        for i in range(0, N):
            normalizedPoints[i][0] = normalizedPoints[i][0] * sX
            normalizedPoints[i][1] = normalizedPoints[i][1] * sY

        T = np.identity(3)
        T[0, 0] = sX
        T[1, 1] = sY
        T[0, 2] = -meanX*sX
        T[1, 2] = -meanY*sY

        return normalizedPoints, T

    
    def checkRT(self, R, t, bInliers):
        bGoods = [False for _ in range(0, len(self.keyPoints1_cv))]
        points3d = [[0.0, 0.0, 0.0] for _ in range(0, len(self.keyPoints1_cv))]
        cosParallaxs = []
        
        # camera matrix P = K[R|t]
        # P1 = K[I|0]
        P1 = np.zeros((3, 4))
        P1[:3, :3] = self.K
        # P2 = K[R|t]
        P2 = np.zeros((3, 4))
        P2[:3, :3] = R
        P2[:, 3] = t
        P2 = self.K @ P2

        # camera orogin
        origin1 = np.zeros(3)
        origin2 = -1.0 * R.transpose() @ t

        # reprojection error threshold
        th2 = self.sigma * self.sigma * 4.0

        nGood = 0

        for i, match in enumerate(self.matches_cv):
            if bInliers[i] is False:
                continue
            
            kp1 = self.keyPoints1_cv[match.queryIdx]
            kp2 = self.keyPoints2_cv[match.trainIdx]

            point3d_C1 = self.triangulate(kp1, kp2, P1, P2) # camera 1 coordinate

            if np.all(np.isfinite(point3d_C1)) is False:
                bGoods[match.queryIdx] = False
                continue

            # check parallax
            normal1 = point3d_C1 - origin1
            dist1 = np.linalg.norm(normal1)
            normal2 = point3d_C1 - origin2
            dist2 = np.linalg.norm(normal2)
            cosParallax = np.dot(normal1, normal2) / (dist1 * dist2)

            # check depth in front of first camera (only if enough parallax, as "infinite" points can easily go to negative depth)
            if point3d_C1[2] <= 0 and cosParallax < 0.99998:
                continue

            # check depth in front of second camera (only if enough parallax, as "infinite" points can easily go to negative depth)
            point3d_C2 = R @ point3d_C1 + t # camera 2 coordinate
            if point3d_C2[2] <= 0 and cosParallax < 0.99998:
                continue

            # check reprojection error in first image
            invZ1 = 1.0 / point3d_C1[2]
            im1x = self.fx * point3d_C1[0] * invZ1 + self.cx
            im1y = self.fy * point3d_C1[1] * invZ1 + self.cy
            squareError1 = (im1x - kp1.pt[0]) * (im1x - kp1.pt[0]) + \
                            (im1y - kp1.pt[1]) * (im1y - kp1.pt[1])
            if squareError1 > th2:
                continue

            # check reprojection error in second image
            invZ2 = 1.0 / point3d_C2[2]
            im2x = self.fx * point3d_C2[0] * invZ2 + self.cx
            im2y = self.fy * point3d_C2[1] * invZ2 + self.cy
            squareError2 = (im2x - kp2.pt[0]) * (im2x - kp2.pt[0]) + \
                            (im2y - kp2.pt[1]) * (im2y - kp2.pt[1])
            if squareError2 > th2:
                continue

            cosParallaxs.append(cosParallax)
            points3d[match.queryIdx] = [point3d_C1[0], point3d_C1[1], point3d_C1[2]]
            nGood += 1

            if cosParallax < 0.99998:
                bGoods[match.queryIdx] = True

        if nGood > 0:
            cosParallaxs.sort()
            idx = min(50, len(cosParallaxs) - 1)
            parallax = math.degrees(math.acos(cosParallaxs[idx]))
        else:
            parallax = 0
        
        return nGood, parallax, points3d, bGoods

    def decomposeE(self, E):
        u, s, vt = svd(E)

        t = u[:, 2]
        t = t / np.linalg.norm(t)

        W = np.zeros((3, 3))
        W[0, 1] = -1.0
        W[1, 0] = 1.0
        W[2, 2] = 1.0

        R1 = u @ W @ vt
        if np.linalg.det(R1) < 0:
            R1 = -1.0 * R1
        
        R2 = u @ W.transpose() @ vt
        if np.linalg.det(R2) < 0:
            R2 = -1.0 * R2

        return t, -t, R1, R2