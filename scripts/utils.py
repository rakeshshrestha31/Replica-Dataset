import numpy as np
from scipy.spatial.transform import Rotation
import cv2


def correct_rotation(R, verbose=False):
    assert(len(R.shape) == 2)
    assert(R.shape[0] == 3)
    assert(R.shape[1] == 3)

    R = R.copy()
    if not is_rotation_righthanded(R, verbose=verbose):
        R[:, 2] = R[:, 2] * -1
        R = R / np.linalg.det(R)


    # preserve direction of z-axis
    angles = Rotation.from_matrix(R).as_euler("xyz", degrees=True)

    if angles[0] > 90:
        rvec = np.array([np.pi, 0., 0.], dtype=np.float32)
        R = R @ cv2.Rodrigues(rvec)[0]

    return R


# https://github.com/intel-isl/Open3D/issues/2206
def is_rotation_righthanded(R, verbose=False):
    assert(len(R.shape) == 2)
    assert(R.shape[0] == 3)
    assert(R.shape[1] == 3)

    res = True

    # Check that axis are right-handed
    if not np.allclose(R[2], np.cross(R[0], R[1])):
        if verbose:
            print("x cross y != z", np.cross(R[0], R[1]), " != ", R[2])
        res = False

    if not np.allclose(R[0], np.cross(R[1], R[2])):
        if verbose:
            print("y cross z != x", np.cross(R[1], R[2]), " != ", R[0])
        res = False

    if not np.allclose(R[1], np.cross(R[2], R[0])):
        if verbose:
            print("z cross x != y", np.cross(R[2], R[0]), " != ", R[1])
        res = False

    # Check that axis' are right-handed
    if not np.allclose(R[:, 2], np.cross(R[:, 0], R[:, 1])):
        if verbose:
            print("x' cross y' != z'", np.cross(R[:, 0], R[:, 1]), " != ", R[:, 2])
        res = False

    if not np.allclose(R[:, 0], np.cross(R[:, 1], R[:, 2])):
        if verbose:
            print("y' cross z' != x'", np.cross(R[:, 1], R[:, 2]), " != ", R[:, 0])
        res = False

    if not np.allclose(R[:, 1], np.cross(R[:, 2], R[:, 0])):
        if verbose:
            print("z' cross x' != y'", np.cross(R[:, 2], R[:, 0]), " != ", R[:, 1])
        res = False

    return res
