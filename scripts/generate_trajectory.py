import sys
from pathlib import Path
from tqdm import  tqdm
import typing
from copy import deepcopy
import re

import open3d as o3d
import cv2
import numpy as np

import sys
sys.path.append(Path(__file__).resolve())
from utils import correct_rotation


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
        rx: typing.Union[np.ndarray, float],
        ry: typing.Union[np.ndarray, float],
        rz: typing.Union[np.ndarray, float],
        theta: typing.Union[np.ndarray, float],
        phi: typing.Union[np.ndarray, float],
) -> np.ndarray:
    """
    z-axis: up (traditionally z)
    y-axis: front (traditionally x)
    x-axis: right (traditionally y)
    """
    # our phi is 0 along y plane (front of the object)
    # but 0 should be along the vertical axis (here z) by default
    phi += np.pi / 2.0
    z = np.asarray(rz * np.cos(phi)).reshape((-1, 1))
    x = np.asarray(rx * np.sin(phi) * np.cos(theta)).reshape((-1, 1))
    y = np.asarray(ry * np.sin(phi) * np.sin(theta)).reshape((-1, 1))
    return np.concatenate((x, y, z), axis=-1)


def generate_trajectory(rx, ry, rz):
    # lookat_point = (4.0, 1., -0.4)
    # lookat_point = (2.75, -1.1, -1.2)

    # looking at origin. Trajectory will be transformed later
    lookat_point = (0.0, 0.0, 0.0)

    # rx = 2.5
    # ry = 1.5
    # rz = 1.3

    # thetas = [np.deg2rad(i) for i in np.linspace(10, 350, 40)]
    # phis = [np.deg2rad(i) for i in np.linspace(-40, 40, 4)]

    thetas = [np.deg2rad(i) for i in np.linspace(10, 350, 26)]
    phis = [0]
    thetas, phis = np.meshgrid(thetas, phis)
    thetas = thetas.reshape((-1,)).tolist()
    phis = phis.reshape((-1,)).tolist()

    poses = []

    # for pose_idx in range(num_poses):
    for theta, phi in zip(thetas, phis):
        # [0, 2 pi]
        theta = (theta % (2 * np.pi))

        # rx_ = rx + (np.random.random() * del_x_max)
        # ry_ = ry + (np.random.random() * del_y_max)
        # rz_ = rz + (np.random.random() * del_z_max)

        cartesian_coords = spherical_to_cartesian(rx, ry, rz, theta, phi)
        cartesian_coords = cartesian_coords.reshape((-1,))

        # (x, y, z) in world coords
        x = cartesian_coords[0] + lookat_point[0]
        y = cartesian_coords[1] + lookat_point[1]
        z = cartesian_coords[2] + lookat_point[2]

        pose = get_camera_pose((x, y, z), lookat_point)
        poses.append(pose)

    del_z = rz / 2.0
    del_zs = [del_z, -del_z]
    num_base_poses = len(poses)

    for del_z in del_zs:
        for i in range(num_base_poses):
            pose = poses[i].copy()
            pose[2, 3] += del_z
            poses.append(pose)

    return poses


def get_extrinsics(roi):
    extrinsics = []
    mesh = o3d.geometry.TriangleMesh()

    extent = roi['extent'].copy()
    extent[0] *= 0.8
    extent[1] *= 1.1

    # look at point slightly higher
    t = roi['centroid']
    t[2] += 0.3

    R = roi['R']

    T = np.eye(4)
    T[:3, :3] = R
    T[:3, -1] = t

    poses = generate_trajectory(*(extent.tolist()))
    poses = [T @ i for i in poses]

    for i, pose in enumerate(poses):
        mesh += (
            o3d.geometry.TriangleMesh.create_coordinate_frame()
            .scale(0.25, [0.0]*3)
            .transform(pose)
        )
        extrinsic = np.linalg.inv(pose)
        extrinsics.append(extrinsic)
        np.savetxt(f'/tmp/extrinsic_{i:06d}.txt', extrinsic.reshape((-1,)))

    # poses = []
    # for i in range(75):
    #     extrinsic = np.loadtxt(f'/tmp/extrinsic_{i:06d}.txt').reshape(4, 4)
    #     pose = np.linalg.inv(extrinsic)
    #     extrinsics.append(extrinsic)
    #     poses.append(pose)
    #     mesh += (
    #         o3d.geometry.TriangleMesh.create_coordinate_frame()
    #         .scale(0.25, [0.0]*3)
    #         .transform(pose)
    #     )

    o3d.io.write_triangle_mesh('/tmp/poses.ply', mesh)

    return extrinsics


def find_scale_matrix(sel_model_file, full_model_file):
    sel_o3d_mesh = o3d.io.read_triangle_mesh(str(sel_model_file), True)
    sel_o3d_mesh.remove_non_manifold_edges()

    full_o3d_mesh = o3d.io.read_point_cloud(str(full_model_file)) # , True)
    # full_o3d_mesh = o3d.io.read_triangle_mesh(str(full_model_file)) # , True)
    # full_o3d_mesh.remove_non_manifold_edges()

    ## sel_mesh for scale
    obb = sel_o3d_mesh.get_oriented_bounding_box()
    R = correct_rotation(obb.R)
    centroid = sel_o3d_mesh.get_center()
    aabb = deepcopy(sel_o3d_mesh).rotate(R.T).get_axis_aligned_bounding_box()
    aabb_extents = aabb.get_extent().reshape((-1,))

    ## full_mesh for scale
    # obb = full_o3d_mesh.get_oriented_bounding_box()
    # R = correct_rotation(obb.R)
    # centroid = full_o3d_mesh.get_center()
    # aabb = deepcopy(full_o3d_mesh).rotate(R.T).get_axis_aligned_bounding_box()
    # aabb_extents = aabb.get_extent().reshape((-1,))

    # add some slack to extent to avoid clipping around the boundary
    extent = np.max(aabb_extents) * 1.2

    scale_matrix = np.eye(4)
    scale_matrix[:3, 3] = centroid
    scale_matrix[:3, :3] = R

    # range [-0.5, 0.5]
    # new_extent = 1.0

    # range [-1.0, 1.0]
    new_extent = 2.0

    scale = extent / new_extent
    scale_matrix[:3, :3] *= scale

    ## sel_mesh for generating trajectory later
    centroid = sel_o3d_mesh.get_center()
    aabb = deepcopy(sel_o3d_mesh).rotate(R.T).get_axis_aligned_bounding_box()
    aabb_extents = aabb.get_extent().reshape((-1,))

    # debug
    inv_scale_matrix = np.linalg.inv(scale_matrix)
    mesh1 = sel_o3d_mesh.transform(inv_scale_matrix)
    mesh2 = full_o3d_mesh.transform(inv_scale_matrix)

    o3d.io.write_triangle_mesh('/tmp/scaled_mesh_sel.ply', mesh1)
    # o3d.io.write_triangle_mesh('/tmp/scaled_mesh_full.ply', mesh2)

    # o3d.io.write_point_cloud('/tmp/scaled_mesh_sel.ply', mesh1)
    o3d.io.write_point_cloud('/tmp/scaled_mesh_full.ply', mesh2)

    print('scale_matrix\n', scale_matrix)

    return {
        'scale_matrix': scale_matrix,
        'R': R,
        'centroid': centroid,
        'extent': aabb_extents
    }


def get_projection_matrix(K, extrinsic):
    proj_mat = K @ extrinsic[:3, :]
    # IDR convention adds [0, 0, 0, 1] row
    proj_mat = np.concatenate((proj_mat, [[0, 0, 0, 1]]), axis=0)
    return proj_mat


def save_idr_format(dataset_root, sel_model_file, full_model_file, K):
    out = find_scale_matrix(sel_model_file, full_model_file)
    scale_matrix = out['scale_matrix']

    extrinsics = get_extrinsics(out)

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

    viz_scaled_cameras(extrinsics, scale_matrix)


def viz_scaled_cameras(extrinsics, scale_matrix):
    mesh = o3d.geometry.TriangleMesh()
    for Tcw in extrinsics:
        Tcw = Tcw @ scale_matrix
        Twc = np.linalg.inv(Tcw)
        mesh += (
            o3d.geometry.TriangleMesh.create_coordinate_frame()
            .scale(0.2, [0.0]*3)
            .transform(Twc)
        )
    o3d.io.write_triangle_mesh('/tmp/scaled_poses.ply', mesh)


def get_num_images(dataset_root):
    rgb_indices = set()
    depth_indices = set()
    pose_indices = set()

    for filepath in dataset_root.iterdir():
        if not filepath.is_file():
            continue
        filename = filepath.name
        if filename.endswith('.jpg'):
            idx = int(filename.split('.')[0])
            rgb_indices.add(idx)

        if filename.startswith('depth') and filename.endswith('.png'):
            idx = int(re.findall(r'\d+', filename)[0])
            depth_indices.add(idx)

        if filename.startswith('pose') and filename.endswith('.txt'):
            idx = int(re.findall(r'\d+', filename)[0])
            pose_indices.add(idx)

    indices = rgb_indices.intersection(depth_indices).intersection(pose_indices)
    return len(indices)


if __name__ == '__main__':
    dataset_root = Path(sys.argv[1])
    sel_model_file = Path(sys.argv[2])
    full_model_file = Path(sys.argv[3])

    intrinsic_file = dataset_root / 'intrinsics.txt'
    fx, fy, cx, cy, width, height, near, far = np.loadtxt(intrinsic_file).tolist()

    K = np.asarray([
        fx, 0, cx,
        0, fy, cy,
        0, 0, 1
   ]).reshape((3, 3))

    save_idr_format(dataset_root, sel_model_file, full_model_file, K)
    exit(0)

    num_images = get_num_images(dataset_root)
    print(f'{num_images=}')
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

        # o3d.io.write_triangle_mesh(f'/tmp/coord_{image_idx}.ply', coord_frame)

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
