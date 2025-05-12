# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.


import os
import sys
sys.path.insert(0, os.getcwd())

import numba
import numpy as np
import utils.transform_utils as T
from scipy.spatial.transform import Rotation as R

from pyquaternion import Quaternion

def pose_euler_to_pose_quaternion(pose_euler):
    pose_quaternion = np.zeros(6)
    pose_quaternion[:3] = pose_euler[:3]
    pose_quaternion[3:] = quaternion_from_euler(*pose_euler[3:])
    return pose_quaternion

def pose_quaternion_to_pose_euler(pose_quaternion):
    pose_euler = np.zeros(6)
    pose_euler[:3] = pose_quaternion[:3]
    pose_euler[3:] = euler_from_quaternion(*pose_quaternion[3:])
    return pose_euler

# ref: https://github.com/stepjam/RLBench/blob/master/rlbench/action_modes/arm_action_modes.py#L30
def calculate_goal_pose(current_pose: np.ndarray, action: np.ndarray):
    a_x, a_y, a_z, a_qx, a_qy, a_qz, a_qw = action
    x, y, z, qx, qy, qz, qw = current_pose

    Qp = Quaternion(a_qw, a_qx, a_qy, a_qz)
    Qch = Quaternion(qw, qx, qy, qz)
    QW = Qp * Qch

    # new_rot = Quaternion(
    #     a_qw, a_qx, a_qy, a_qz) * Quaternion(qw, qx, qy, qz)
    qw, qx, qy, qz = list(QW)
    pose = [a_x + x, a_y + y, a_z + z] + [qx, qy, qz, qw]
    return np.array(pose)

# ref: https://math.stackexchange.com/questions/2124361/quaternions-multiplication-order-to-rotate-unrotate
def calculate_action(current_pose: np.ndarray, goal_pose: np.ndarray):
    x1, y1, z1, qx1, qy1, qz1, qw1 = current_pose
    x2, y2, z2, qx2, qy2, qz2, qw2 = goal_pose
    
    #QW == Qp * Qch
    QW = Quaternion(qw2, qx2, qy2, qz2)
    Qch = Quaternion(qw1, qx1, qy1, qz1)
    # Qp == QW * Qch.Inversed
    Qp = QW * Qch.inverse

    # QW = R.from_quat([qx2, qy2, qz2, qw2])
    # Qch = R.from_quat([qx1, qy1, qz1, qw1])
    # Qp = QW * Qch.inv
    # Qp = Qp.as_quat()

    a_qw, a_qx, a_qy, a_qz = list(Qp)
    a_x, a_y, a_z = x2-x1, y2-y1, z2-z1

    action = [a_x, a_y, a_z] + [a_qx, a_qy, a_qz, a_qw]
    return np.array(action)

def compute_rel_transform(A_pos, A_mat, B_pos, B_mat):
    T_WA = np.vstack((np.hstack((A_mat, A_pos[:, None])), [0, 0, 0, 1]))
    T_WB = np.vstack((np.hstack((B_mat, B_pos[:, None])), [0, 0, 0, 1]))

    T_AB = np.matmul(np.linalg.inv(T_WA), T_WB)

    return T_AB[:3, 3], T_AB[:3, :3]

# target_obj_pose, grasp_obj_pose
def get_rel_pose(pose1, pose2):
    pos1 = np.array(pose1[:3])
    quat1 = np.array(pose1[3:])
    mat1 = T.quat2mat(quat1)
    pos2 = np.array(pose2[:3])
    quat2 = np.array(pose2[3:])
    mat2 = T.quat2mat(quat2)

    pos, mat = compute_rel_transform(pos1, mat1, pos2, mat2)
    quat = T.mat2quat(mat)
    return np.concatenate([pos, quat])


# def realtive_to_target_to_world(env, obj_pose_relative_to_cab, cabinet):
def relative_to_target_to_world(subgoal_relative_to_target, target_obj_pose):
    pos1 = subgoal_relative_to_target[:3]
    mat1 = T.quat2mat(subgoal_relative_to_target[3:])
    pos2 = target_obj_pose[:3]
    mat2 = T.quat2mat(target_obj_pose[3:])

    # T_WA = T_WB @ T_BA
    T_BA = np.vstack((np.hstack((mat1, pos1[:, None])), [0, 0, 0, 1]))
    T_WB = np.vstack((np.hstack((mat2, pos2[:, None])), [0, 0, 0, 1]))
    T_WA = np.matmul(T_WB, T_BA)

    pos = T_WA[:3, 3]
    mat = T_WA[:3, :3]
    quat = T.mat2quat(mat)
    return np.concatenate([pos, quat])

def euler_from_quaternion(x, y, z, w):
    # import math
    # t0 = +2.0 * (w * x + y * z)
    # t1 = +1.0 - 2.0 * (x * x + y * y)
    # roll_x = math.atan2(t0, t1)

    # t2 = +2.0 * (w * y - z * x)
    # t2 = +1.0 if t2 > +1.0 else t2
    # t2 = -1.0 if t2 < -1.0 else t2
    # pitch_y = math.asin(t2)

    # t3 = +2.0 * (w * z + x * y)
    # t4 = +1.0 - 2.0 * (y * y + z * z)
    # yaw_z = math.atan2(t3, t4)

    # return roll_x, pitch_y, yaw_z

    rot = R.from_quat([x, y, z, w]) # (x, y, z, w)
    euler = rot.as_euler('xyz')
    return euler

# def matrix_from_quaternion(x, y, z, w):
#     rot = R.from_quat([x, y, z, w]) # (x, y, z, w)
#     mat = rot.as_matrix()
#     return mat

def quaternion_from_euler(euler):
    rot = R.from_euler('xyz', euler) 
    quat = rot.as_quat()    # (x, y, z, w)
    return quat


def rodrigues(r, calculate_jacobian=True):
    """Computes the Rodrigues transform and its derivative

    :param r: either a 3-vector representing the rotation parameter, or a full rotation matrix
    :param calculate_jacobian: indicates if the Jacobian of the transform is also required
    :returns: If `calculate_jacobian` is `True`, the Jacobian is given as the second element of the returned tuple.
    """

    r = np.array(r, dtype=np.double)
    eps = np.finfo(np.double).eps

    if np.all(r.shape == (3, 1)) or np.all(r.shape == (1, 3)) or np.all(r.shape == (3,)):
        r = r.flatten()
        theta = np.linalg.norm(r)
        if theta < eps:
            r_out = np.eye(3)
            if calculate_jacobian:
                jac = np.zeros((3, 9))
                jac[0, 5] = jac[1, 6] = jac[2, 1] = -1
                jac[0, 7] = jac[1, 2] = jac[2, 3] = 1

        else:
            c = np.cos(theta)
            s = np.sin(theta)
            c1 = 1. - c
            itheta = 1.0 if theta == 0.0 else 1.0 / theta
            r *= itheta
            I = np.eye(3)
            rrt = np.array([r * r[0], r * r[1], r * r[2]])
            _r_x_ = np.array([[0, -r[2], r[1]], [r[2], 0, -r[0]], [-r[1], r[0], 0]])
            r_out = c * I + c1 * rrt + s * _r_x_
            if calculate_jacobian:
                drrt = np.array([[r[0] + r[0], r[1], r[2], r[1], 0, 0, r[2], 0, 0],
                                 [0, r[0], 0, r[0], r[1] + r[1], r[2], 0, r[2], 0],
                                 [0, 0, r[0], 0, 0, r[1], r[0], r[1], r[2] + r[2]]])
                d_r_x_ = np.array([[0, 0, 0, 0, 0, -1, 0, 1, 0],
                                   [0, 0, 1, 0, 0, 0, -1, 0, 0],
                                   [0, -1, 0, 1, 0, 0, 0, 0, 0]])
                I = np.array([I.flatten(), I.flatten(), I.flatten()])
                ri = np.array([[r[0]], [r[1]], [r[2]]])
                a0 = -s * ri
                a1 = (s - 2 * c1 * itheta) * ri
                a2 = np.ones((3, 1)) * c1 * itheta
                a3 = (c - s * itheta) * ri
                a4 = np.ones((3, 1)) * s * itheta
                jac = a0 * I + a1 * rrt.flatten() + a2 * drrt + a3 * _r_x_.flatten() + a4 * d_r_x_
    elif np.all(r.shape == (3, 3)):
        u, d, v = np.linalg.svd(r)
        r = np.dot(u, v)
        rx = r[2, 1] - r[1, 2]
        ry = r[0, 2] - r[2, 0]
        rz = r[1, 0] - r[0, 1]
        s = np.linalg.norm(np.array([rx, ry, rz])) * np.sqrt(0.25)
        c = np.clip((np.sum(np.diag(r)) - 1) * 0.5, -1, 1)
        theta = np.arccos(c)
        if s < 1e-5:
            if c > 0:
                r_out = np.zeros((3, 1))
            else:
                rx, ry, rz = np.clip(np.sqrt((np.diag(r) + 1) * 0.5), 0, np.inf)
                if r[0, 1] < 0:
                    ry = -ry
                if r[0, 2] < 0:
                    rz = -rz
                if np.abs(rx) < np.abs(ry) and np.abs(rx) < np.abs(rz) and ((r[1, 2] > 0) != (ry * rz > 0)):
                    rz = -rz

                r_out = np.array([[rx, ry, rz]]).T
                theta /= np.linalg.norm(r_out)
                r_out *= theta
            if calculate_jacobian:
                jac = np.zeros((9, 3))
                if c > 0:
                    jac[1, 2] = jac[5, 0] = jac[6, 1] = -0.5
                    jac[2, 1] = jac[3, 2] = jac[7, 0] = 0.5
        else:
            vth = 1.0 / (2.0 * s)
            if calculate_jacobian:
                dtheta_dtr = -1. / s
                dvth_dtheta = -vth * c / s
                d1 = 0.5 * dvth_dtheta * dtheta_dtr
                d2 = 0.5 * dtheta_dtr
                dvardR = np.array([
                    [0, 0, 0, 0, 0, 1, 0, -1, 0],
                    [0, 0, -1, 0, 0, 0, 1, 0, 0],
                    [0, 1, 0, -1, 0, 0, 0, 0, 0],
                    [d1, 0, 0, 0, d1, 0, 0, 0, d1],
                    [d2, 0, 0, 0, d2, 0, 0, 0, d2]])
                dvar2dvar = np.array([
                    [vth, 0, 0, rx, 0],
                    [0, vth, 0, ry, 0],
                    [0, 0, vth, rz, 0],
                    [0, 0, 0, 0, 1]])
                domegadvar2 = np.array([
                    [theta, 0, 0, rx * vth],
                    [0, theta, 0, ry * vth],
                    [0, 0, theta, rz * vth]])
                jac = np.dot(np.dot(domegadvar2, dvar2dvar), dvardR)
                for ii in range(3):
                    jac[ii] = jac[ii].reshape((3, 3)).T.flatten()
                jac = jac.T
            vth *= theta
            r_out = np.array([[rx, ry, rz]]).T * vth
    else:
        raise Exception("rodrigues: input matrix must be 1x3, 3x1 or 3x3.")
    if calculate_jacobian:
        return r_out, jac
    else:
        return r_out


def rodrigues2rotmat(r):
    # R = np.zeros((3, 3))
    r_skew = np.array([[0, -r[2], r[1]], [r[2], 0, -r[0]], [-r[1], r[0], 0]])
    theta = np.linalg.norm(r)
    return np.identity(3) + np.sin(theta) * r_skew + (1 - np.cos(theta)) * r_skew.dot(r_skew)


if __name__ == "__main__":
    target_obj_pose = np.array([ 0.32260922, -0.10839751,  0.96184993,  0.57075304,  0.05685034, -0.81511486,  0.08122107])
    grasp_obj_pose = np.array([ 1.99161470e-01,  3.34810495e-01,  7.91927218e-01, -1.26936930e-05,  2.39512883e-06,  9.92952466e-01, -1.18513443e-01])
    # obj_pose_relative_to_target = np.array([ 0.17114308, -0.45885825, -0.02656152, -0.01118924, -0.57345784,  0.0159555,  0.81900305])

    np.set_printoptions(precision=3)
    obj_pose_relative_to_target = get_rel_pose(target_obj_pose, grasp_obj_pose)
    grasp_obj_pose_2 = relative_to_target_to_world(obj_pose_relative_to_target, target_obj_pose)
    print("grasp_obj_pose", grasp_obj_pose)
    print("grasp_obj_pose_2", grasp_obj_pose_2)

    obj_pose_relative_to_target = get_rel_pose(target_obj_pose, grasp_obj_pose)
    grasp_obj_pose_2 = relative_to_target_to_world(obj_pose_relative_to_target, target_obj_pose)
    print("grasp_obj_pose", grasp_obj_pose)
    print("grasp_obj_pose_2", grasp_obj_pose_2)