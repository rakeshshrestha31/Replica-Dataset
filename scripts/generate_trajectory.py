import sys
from pathlib import Path
from tqdm import  tqdm
import typing

import open3d as o3d
import cv2
import numpy as np


def get_camera_pose(eye, center, up=[0, 0, 1]):
    """adapted from trescope.blender.blender_front3d.setCamera
    z-axis points forward (unlike the orignal/OpenGL convention)
    """
    eye = np.array(list(eye))
    center = np.array(list(center))
    north = np.array(list(up))
    direction = center - eye
    forward = direction / np.linalg.norm(direction)
    right = np.cross(-north, forward)
    up = np.cross(forward, right)
    rotation = np.vstack([right, up, forward]).T
    matrix = np.eye(4)
    matrix[:3, :3] = rotation
    matrix[:3, -1] = eye

    return matrix


def spherical_to_cartesian(
        r: typing.Union[np.ndarray, float],
        theta: typing.Union[np.ndarray, float],
        phi: typing.Union[np.ndarray, float],
) -> np.ndarray:
    """
    z-axis: up (traditionally z)
    y-axis: front (traditionally x)
    x-axis: right (traditionally y)
    """
    # our phi is 0 along z plane (front of the object)
    # but 0 should be along the vertical axis (here y) by default
    phi += np.pi / 2.0
    # y here is the the vertical axis (not z)
    z = np.asarray(r * np.cos(phi)).reshape((-1, 1))
    x = np.asarray(r * np.sin(phi) * np.cos(theta)).reshape((-1, 1))
    y = np.asarray(r * np.sin(phi) * np.sin(theta)).reshape((-1, 1))
    return np.concatenate((x, y, z), axis=-1)


def generate_trajectory(num_poses=64):
    # lookat_point = (4.0, 1., -0.4)
    lookat_point = (3.25, -1.3, -1.5)

    max_dist_to_object = 2.75
    min_dist_to_object = 0.75

    max_phi_to_object = np.deg2rad(45)
    del_theta_min = np.deg2rad(15)
    del_theta_max = np.deg2rad(45)

    theta = 0.0
    poses = []

    for pose_idx in range(num_poses):
        del_theta = np.random.uniform(del_theta_min, del_theta_max)
        theta += del_theta

        # [0, 2 pi]
        theta = (theta % (2 * np.pi))

        # phi = np.random.uniform(0.0, 1.0) * max_phi_to_object
        # r = np.random.random() * max_dist_to_object
        # cartesian_coords = spherical_to_cartesian(r=r, theta=theta, phi=phi)
        # cartesian_coords = cartesian_coords.reshape((-1,))
        # assert(cartesian_coords.shape[0] == 3)

        r1 = 2.1 + (np.random.random() * 0.1)
        r2 = 1.2 + (np.random.random() * 0.015)
        z = 0.5 + (np.random.random() * 0.5)

        cartesian_coords = [
            r1 * np.cos(theta),
            r2 * np.sin(theta),
            z
        ]

        # (x, y, z) in world coords
        x = cartesian_coords[0] + lookat_point[0]
        y = cartesian_coords[1] + lookat_point[1]
        z = cartesian_coords[2] + lookat_point[2]

        poses.append(get_camera_pose((x, y, z), lookat_point))

    return poses


def get_extrinsics():
    extrinsics = []

    poses = generate_trajectory()
    mesh = o3d.geometry.TriangleMesh()
    for i, pose in enumerate(poses):
        mesh += (
            o3d.geometry.TriangleMesh.create_coordinate_frame()
            .scale(0.25, [0.0]*3)
            .transform(pose)
        )
        extrinsic = np.linalg.inv(pose)
        extrinsics.append(extrinsic)
        np.savetxt(f'/tmp/extrinsic_{i:06d}.txt', extrinsic.reshape((-1,)))
    o3d.io.write_triangle_mesh('/tmp/poses.ply', mesh)

    # for i in range(64):
    #     extrinsics.append(np.loadtxt(f'/tmp/extrinsic_{i:06d}.txt').reshape(4, 4))

    return extrinsics


def find_scale_matrix(model_file):
    o3d_mesh = o3d.io.read_triangle_mesh(str(model_file), True)
    # o3d_mesh = o3d.io.read_point_cloud(str(model_file))
    o3d_mesh.remove_non_manifold_edges()

    centroid = o3d_mesh.get_center()
    aabb = o3d_mesh.get_axis_aligned_bounding_box()
    # add some slack to extent to avoid clipping around the boundary
    extent = np.max(aabb.get_extent().reshape((-1,))) # * 1.2

    scale_matrix = np.eye(4)
    scale_matrix[:3, 3] = centroid

    # range [-0.5, 0.5]
    # new_extent = 1.0

    # range [-1.0, 1.0]
    new_extent = 2.0

    scale_matrix[0, 0] = scale_matrix[1, 1] = scale_matrix[2, 2] = extent / new_extent

    print('scale_matrix\n', scale_matrix)
    return scale_matrix


def get_projection_matrix(K, extrinsic):
    proj_mat = K @ extrinsic[:3, :]
    # IDR convention adds [0, 0, 0, 1] row
    proj_mat = np.concatenate((proj_mat, [[0, 0, 0, 1]]), axis=0)
    return proj_mat


def save_idr_format(dataset_root, model_file, K):
    extrinsics = get_extrinsics()
    scale_matrix = find_scale_matrix(model_file)

    proj_matrices = [
        get_projection_matrix(K, extrinsic)
        for extrinsic in extrinsics
    ]

    cameras = [
        [
            (f'world_mat_{img_id}', proj_matrices[img_id]),
            (f'camera_mat_{img_id}', K),
            (f'scale_mat_{img_id}', scale_matrix),
            (f'extrinsic_{img_id}', extrinsics[img_id]),
        ]
        for img_id in range(len(extrinsics))
    ]
    cameras = dict([j for i in cameras for j in i])

    cameras_file = '/tmp/cameras.npz'
    np.savez(str(cameras_file), **cameras)


if __name__ == '__main__':
    dataset_root = Path(sys.argv[1])
    model_file = Path(sys.argv[2])

    intrinsic_file = dataset_root / 'intrinsics.txt'
    fx, fy, cx, cy, width, height, near, far = np.loadtxt(intrinsic_file).tolist()

    K = np.asarray([
        fx, 0, cx,
        0, fy, cy,
        0, 0, 1
   ]).reshape((3, 3))

    save_idr_format(dataset_root, model_file, K)
    exit(0)

    num_images = 64
    depth_scale = 65535.0 * 0.1

    intrinsics = o3d.camera.PinholeCameraIntrinsic(
        int(width), int(height), fx, fy, cx, cy
    )

    for image_idx in tqdm(range(num_images)):
        # if image_idx != 47:
        #     continue
        depth_file = dataset_root / f'depth{image_idx:06d}.png'
        rgb_file = dataset_root / f'{image_idx:06d}.jpg'
        pose_file = dataset_root / f'pose{image_idx:06d}.txt'

        T_cam_world = np.loadtxt(pose_file)
        T_world_cam = np.linalg.inv(T_cam_world)

        coord_frame = (
            o3d.geometry.TriangleMesh.create_coordinate_frame()
            .scale(0.25, [0.0]*3)
            .transform(T_world_cam)
        )

        o3d.io.write_triangle_mesh(f'/tmp/coord_{image_idx}.ply', coord_frame)

        depth = (
            cv2.imread(str(depth_file), cv2.IMREAD_ANYDEPTH)
            .astype(np.float32) / depth_scale
        )

        mask = (depth > 0).astype(np.uint8) * 255
        mask_dir = dataset_root / 'mask'
        mask_dir.mkdir(exist_ok=True)
        cv2.imwrite(str(mask_dir / f'{image_idx:06d}.png'), mask)
        continue

        mask = np.zeros_like(depth)
        # mask[663:721, 227:593] = 1.0
        mask[645:750, 200:610] = 1.0
        depth *= mask

        rgb = (
            cv2.imread(str(rgb_file), -1)[..., ::-1]
            .copy().astype(np.float32) / 255.
        )

        depth = o3d.geometry.Image(depth)
        rgb = o3d.geometry.Image(rgb)
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            rgb, depth, depth_scale=1.0, depth_trunc=7.0
        )

        pcd = (
            o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsics)
            .transform(T_world_cam)
        )

        o3d.io.write_point_cloud(f'/tmp/depth_{image_idx}.ply', pcd)

        if image_idx > 3:
            break
