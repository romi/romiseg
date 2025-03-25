#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 26 16:53:46 2019

@author: alienor
"""
import logging

import numpy as np
import torch

logger = logging.getLogger(__file__)


def avoid_eps(a, eps):
    """Sets array elements to zero within a specified epsilon threshold.

    This function modifies the input tensor in place. It evaluates the absolute
    values of the elements and sets those which are below the epsilon value to
    zero. It is intended for filtering out very small values in numerical arrays,
    helping to avoid computational inaccuracies due to tiny floating-point values.

    Parameters
    ----------
    a : torch.Tensor
        The input tensor whose elements will be processed. This tensor is modified
        in place by the function.
    eps : float
        The epsilon threshold. Elements in the tensor with absolute values smaller
        than this value will be set to zero.

    Returns
    -------
    torch.Tensor
        The modified input tensor `a`, where elements with absolute values smaller
        than the epsilon have been set to zero.
    """
    a[torch.abs(a) < eps] = 0
    return a


def basis_vox(min_vec, w, h, l):
    """Generates a grid of voxels with specified dimensions and starting position.

    This function creates a grid of 3D points (voxels) with predefined width, height,
    and length originating from a specific minimum position in 3D space. The
    function returns an array where each row represents a voxel with X, Y, Z
    coordinates and an additional empty fourth column.

    Parameters
    ----------
    min_vec : array-like
        The minimum position in 3D space as a vector of length 3 (minx, miny, minz).
    w : int
        The width of the voxel grid.
    h : int
        The height of the voxel grid.
    l : int
        The length of the voxel grid.

    Returns
    -------
    numpy.ndarray
        A (w*h*l, 4) array where each row consists of [X, Y, Z, 0], representing
        voxel coordinates in 3D space followed by a placeholder value.
    """
    # Unpack minimum coordinates for the voxel grid.
    minx, miny, minz = min_vec

    # Generate evenly spaced X, Y, Z coordinates based on width, height, and length.
    # X changes along the width with `w` points.
    X = np.linspace(minx, minx + w - 1, w)
    # Y changes along the height with `h` points.
    Y = np.linspace(miny, miny + h - 1, h)
    # Z changes along the length with `l` points.
    Z = np.linspace(minz, minz + l - 1, l)

    # Initialize a (w*h*l, 4) array for voxels with an extra 4th column as a placeholder.
    voxels = np.zeros((w * h * l, 4))
    # Set the X coordinate for each voxel by replicating X for all layers.
    voxels[:, 0] = np.repeat(X, h * l)
    # Repeat Y values along the length dimension and tile them to cover width.
    Y_bis = np.repeat(Y, l)
    voxels[:, 1] = np.tile(Y_bis, w)  # Tile the repeated Y values for all width layers.
    # Tile Z values to cover the entire grid across width and height.
    voxels[:, 2] = np.tile(Z, h * w)

    return voxels


def basis_vox_pipeline(min_vec, max_vec, w, h, l):
    """Generate a voxel representation within the specified 3D bounding box.

    This function creates a voxel grid based on specified dimensions (width,
    height, depth) within a 3-dimensional bounding box defined by the minimum
    and maximum coordinates. The spatial locations are computed using evenly
    spaced points in each dimension, and an array containing the voxel
    coordinates is returned.

    Parameters
    ----------
    min_vec : Tuple[float, float, float]
        The minimum (x, y, z) values of the bounding box within which the voxels are generated.
    max_vec : Tuple[float, float, float]
        The maximum (x, y, z) values of the bounding box within which the voxelsare generated.
    w : int
        The number of divisions (points) along the x-axis inside the bounding box.
    h : int
        The number of divisions (points) along the y-axis inside the bounding box.
    l : int
        The number of divisions (points) along the z-axis inside the bounding box.

    Returns
    -------
    numpy.ndarray
        A 2D array of shape `(w * h * l, 4)` where each row represents a single voxel.
        The first three columns contain the (x, y, z) coordinates of the
        voxel, while the fourth column remains initialized to zero.
    """
    # Unpack bounding box boundaries (min and max values for each axis)
    minx, miny, minz = min_vec
    maxx, maxy, maxz = max_vec

    # Create evenly spaced coordinates for each axis based on the given resolution
    X = np.linspace(minx, maxx, w)  # Points along the x-axis
    Y = np.linspace(miny, maxy, h)  # Points along the y-axis
    Z = np.linspace(minz, maxz, l)  # Points along the z-axis

    # Initialize the voxel array: each row stores (x, y, z, 0)
    # Total rows = w * h * l, representing number of voxels
    voxels = np.zeros((w * h * l, 4))  # Fourth column is unused (set to 0)

    # Set x-coordinates: repeat each x-coordinate for all (y, z) combinations
    voxels[:, 0] = np.repeat(X, h * l)
    # Set y-coordinates: repeat each y-coordinate for the depth (z), then tile for x
    Y_bis = np.repeat(Y, l)  # Repeat each Y value for all z-coordinates
    voxels[:, 1] = np.tile(Y_bis, w)  # Tile the repeated Y values for all x
    # Set z-coordinates: tile entire z-coordinate list for all (x, y) combinations
    voxels[:, 2] = np.tile(Z, h * w)

    return voxels


def get_int(fx, fy, cx, cy):
    """Constructs a camera intrinsic matrix based on the provided focal lengths and principal point coordinates.

    The function generates a 3x3 tensor representing the intrinsic
    parameter matrix. This representation is commonly used in computer
    vision tasks involving camera calibration or projection modeling.
    The intrinsic matrix takes the form:

        [[fx,  0, cx],
         [ 0, fy, cy],
         [ 0,  0,  1]]

    Parameters
    ----------
    fx : float
        The focal length of the camera in the x-direction.
    fy : float
        The focal length of the camera in the y-direction.
    cx : float
        The x-coordinate of the principal point.
    cy : float
        The y-coordinate of the principal point.

    Returns
    -------
    Tensor
        A 3x3 tensor representing the camera's intrinsic matrix.
    """
    intri = torch.tensor([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    return intri


def get_extr(rx, ry, rz, x, y, z):
    """Computes the extrinsic camera matrix based on given rotation angles and translation coordinates.

    The function forms a 3x4 matrix that represents the combined rotation and translation in 3D space.

    The method calculates the rotation matrices for the x, y, and z axes based on the
    provided angles. It ensures numerical stability by avoiding extremely small values
    through a custom `avoid_eps` function. These matrices are combined to form a
    composite rotation matrix. The translation part is computed and integrated with the
    rotation to form the final extrinsic matrix.

    Parameters
    ----------
    rx : float
        Rotation angle (in radians) around the x-axis.
    ry : float
        Rotation angle (in radians) around the y-axis.
    rz : float
        Rotation angle (in radians) around the z-axis.
    x : float
        Translation along the x-axis.
    y : float
        Translation along the y-axis.
    z : float
        Translation along the z-axis.

    Returns
    -------
    torch.Tensor
        A 3x4 extrinsic matrix representing the rotation and translation in 3D space.
    """
    # Initialize a 3x4 matrix to store the rotation and translation components
    rt = torch.zeros((3, 4))

    # Create and populate the 3x3 rotation matrix for rotation around the x-axis
    rotx = torch.zeros((3, 3))
    c = np.cos(rx)  # Cosine of the rotation angle
    s = np.sin(rx)  # Sine of the rotation angle
    rotx[0, 0] = 1  # Fixed axis for x rotation
    rotx[2, 2] = c  # Rotate z coordinates around the x-axis
    rotx[1, 2] = -s
    rotx[2, 1] = s
    rotx[1, 1] = c

    # Create and populate the 3x3 rotation matrix for rotation around the y-axis
    roty = torch.zeros((3, 3))
    c = np.cos(ry)
    s = np.sin(ry)
    roty[0, 0] = c  # Rotate x coordinates around the y-axis
    roty[2, 2] = c
    roty[0, 2] = s
    roty[2, 0] = -s
    roty[1, 1] = 1  # Fixed axis for y rotation

    # Create and populate the 3x3 rotation matrix for rotation around the z-axis
    rotz = torch.zeros((3, 3))
    c = np.cos(rz)
    s = np.sin(rz)
    rotz[0, 0] = c  # Rotate x and y coordinates around the z-axis
    rotz[1, 1] = c
    rotz[0, 1] = -s
    rotz[1, 0] = s
    rotz[2, 2] = 1  # Fixed axis for z rotation

    # Add a small value (eps) to avoid numerical issues with near-zero values
    eps = 1e-7
    rotz = avoid_eps(rotz, eps)
    roty = avoid_eps(roty, eps)
    rotx = avoid_eps(rotx, eps)

    # Debugging: Display intermediate rotation matrices
    logger.debug(f"Rotation matrix for x-axis (rotx):\n{rotx}")
    logger.debug(f"Rotation matrix for y-axis (roty):\n{roty}")
    logger.debug(f"Rotation matrix for z-axis (rotz):\n{rotz}")

    # Combined rotation matrix by multiplying rotations in x, y, and z order
    rot = torch.mm(torch.mm(rotx, roty), rotz)

    # Create a column vector for translation
    trans = torch.tensor([[x], [y], [z]])

    # Adjust translation based on the rotation, effectively negating it
    trans = -torch.mm(torch.transpose(rotz, 0, 1), trans)

    # Populate the extrinsic matrix with the rotation and translation components
    rt[:3, :3] = rot  # Top-left 3x3 block for rotation
    rt[:, 3] = torch.transpose(trans, 0, 1)  # Rightmost column for translation

    return rt


def get_trajectory(N_cam, x0, y0, z0, rx, ry):
    """
    Calculates the trajectory of cameras based on specified parameters and returns
    a tensor of extrinsic camera matrices for 3D transformations.

    Each camera position, derived from the provided parameters, is calculated along
    a circular trajectory. The extrinsic matrix for each position is then computed
    using the `get_extr` method which specifies the camera's rotation and
    translation in 3D space.

    Parameters
    ----------
    N_cam : int
        The number of cameras to position on the circular trajectory.
    x0 : float
        The radius of the circle along the x-axis where the cameras are
        positioned.
    y0 : float
        Value used in calculating specific positions of cameras along the circular
        path.
    z0 : float
        The height (z-coordinate) of the cameras from ground level.
    rx : float
        The rotation around the x-axis for all cameras.
    ry : float
        The rotation around the y-axis for all cameras.

    Returns
    -------
    torch.Tensor
        A tensor of shape (N_cam, 3, 4) containing the extrinsic matrices for the
        cameras. Each extrinsic matrix describes the position and rotation of a
        camera in 3D space.

    """
    d_theta = -2 * np.pi / N_cam
    extrinsics = torch.zeros(N_cam, 3, 4)
    for i in range(N_cam):
        x1 = x0 * np.cos(i * d_theta)  # x pos of camera
        y1 = x0 * np.sin(i * d_theta)  # y pos of camera
        rz = d_theta * i + np.pi / 2  # camera pan
        pose = get_extr(rx, ry, rz, x1, y1, z0)
        extrinsics[i, :, :] = pose
    return extrinsics


def project_coordinates(torch_voxels, intrinsics, extrinsics, give_prod):
    """Projects 3D voxel coordinates into a 2D camera image plane using the given camera intrinsics and extrinsics.

    Additionally, returns pre-projection coordinates if specified.

    Parameters
    ----------
    torch_voxels : torch.Tensor
        A tensor representing 3D voxels in homogeneous coordinates.
    intrinsics : torch.Tensor
        Intrinsic matrix of the camera.
    extrinsics : torch.Tensor
        Extrinsic matrix of the camera, containing rotation and translation
        parameters for transforming the 3D coordinates into the camera's
        coordinate system.
    give_prod : bool
        If True, returns the intermediate 3D coordinates in the camera's
        coordinate frame before applying projection to the 2D image plane.

    Returns
    -------
    torch.Tensor or Tuple[torch.Tensor, torch.Tensor]
        If `give_prod` is False, returns only the projected 2D coordinates
        (xy_coords) as a tensor. If `give_prod` is True, returns a tuple
        containing the pre-projection 3D coordinates (prod1) and the projected
        2D coordinates (xy_coords).
    """
    torch_voxels = torch_voxels.unsqueeze(0)  # homogeneous coordinates
    t = torch_voxels.clone()
    # t = t-torch.mean(t, dim = 1)
    t[:, :, 3] = 1

    t = t.permute(0, 2, 1)  # convenient for matrix product
    ext = extrinsics[:, 0:3, :]  # several camera poses
    prod = torch.matmul(ext.double(), t)  # coordinate change
    if give_prod == True:
        prod1 = prod.clone()
    prod[:, 0, :] = prod[:, 0, :] / prod[:, 2, :]
    prod[:, 1, :] = prod[:, 1, :] / prod[:, 2, :]
    prod[:, 2, :] = prod[:, 2, :] / prod[:, 2, :]
    xy_coords = prod[:, 0:3, :]
    xy_coords = torch.matmul(intrinsics.double(), xy_coords)  # coordinate change
    if give_prod == True:
        return prod1, xy_coords
    else:
        return xy_coords


def correct_coords_outside(coordinates, Sx, Sy, xinit, yinit, val):
    """Corrects coordinates that are outside specific bounds by assigning them a fixed value.

    This function modifies coordinate values that fall outside the defined
    rectangular bounds determined by the given parameters. The adjustment
    is applied element-wise, where the input tensor is updated if its coordinates
    violate the bounds. Corrected coordinates are cast to integer type before
    returning.

    Parameters
    ----------
    coordinates : torch.Tensor
        A 3D tensor containing coordinate values. The first dimension is for
        indexing over examples, the second dimension contains x and y coordinates,
        and the third dimension represents other features.
    Sx : float
        Width of the bounding box along the x-axis.
    Sy : float
        Height of the bounding box along the y-axis.
    xinit : float
        Initial x-coordinate around which bounds are defined.
    yinit : float
        Initial y-coordinate around which bounds are defined.
    val : float
        The value assigned to the coordinates that fall outside the bounding
        rectangle.

    Returns
    -------
    torch.Tensor
        A tensor of integers with the same shape as the input containing corrected
        coordinates. Values outside the bounds are replaced by the specified `val`.
    """

    coords = coordinates.clone()

    indices = (coords[:, 0, :] < (xinit - Sx) / 2)  # lower bound for x
    ind_stack = torch.stack([indices] * 3, dim=1)
    coords[ind_stack] = val

    indices = (coords[:, 1, :] < (yinit - Sy) / 2)  # lower bound for y
    ind_stack = torch.stack([indices] * 3, dim=1)
    coords[ind_stack] = val

    indices = (coords[:, 0, :] > (xinit + Sx) / 2)  # upper bound for x
    ind_stack = torch.stack([indices] * 3, dim=1)
    coords[ind_stack] = val

    indices = (coords[:, 1, :] > (yinit + Sy) / 2)  # upper bound for y
    ind_stack = torch.stack([indices] * 3, dim=1)
    coords[ind_stack] = val

    return coords.long()


def adjust_predictions(preds):
    """
    Adjusts a 4D tensor of predictions by adding a label class for voxels projected
    outside the image boundary and ensures that the tensor is properly flattened
    with an additional label for outside voxels.

    Parameters
    ----------
    preds : torch.Tensor
        A 4D tensor of shape (batch_size, channels, height, width) representing
        the predictions for each voxel.

    Returns
    -------
    torch.Tensor
        A 2D tensor where the predictions are flattened along all dimensions
        except the batch dimension, with an additional row corresponding to
        voxels projected outside the image.
    """
    outside_proj_label = (preds[:, :, :, 0] * 0).unsqueeze(-1)
    preds = torch.cat([preds, outside_proj_label], dim=3)  # Add a label class: voxel projects outside image
    preds_flat = torch.flatten(preds, end_dim=-2)  # Flatten the predictions

    outside_label = preds_flat[0] * 0
    outside_label[-1] = 1
    outside_label = outside_label.unsqueeze(0)
    preds_flat = torch.cat([preds_flat, outside_label])  # Add a last prediction where all
    # voxels that project outside the  image will collect their class

    return preds_flat


def flatten_coordinates(coords, shape_predictions):
    """Flatten coordinates based on provided shape predictions.

    This function processes the coordinates by mapping them to flattened indices
    using the shape_predictions parameters. It effectively performs a manual
    ravel_multi_index operation, with additional handling to set negative indexes
    to -1.

    Parameters
    ----------
    coords : torch.Tensor
        Tensor of shape (N, 2) representing the input coordinates. Each row
        contains [x, y] values for a specific point.
    shape_predictions : torch.Tensor
        Tensor containing the predictions for shape values. This must include
        three elements that define the dimensions along which the transformation
        is applied.

    Returns
    -------
    torch.Tensor
        A 1D tensor containing flattened indices created from the input
        coordinates and predictions. Flattened indices that correspond to
        invalid input are set to -1.
    """
    xx = coords[:, 0]
    yy = coords[:, 1]
    xy = torch.mul(xx, shape_predictions[2]) + yy  # manually perform ravel_multi_index from numpy, along X and Y

    flat_factor = shape_predictions[1] * shape_predictions[2]
    flat_vec = torch.mul(torch.linspace(0, shape_predictions[0] - torch.tensor(1), shape_predictions[0]), flat_factor)
    flat_vec = flat_vec.unsqueeze(1).long()
    flat_coo = torch.add(xy, flat_vec)  # Perform it along the views N_cam
    flat_coo[xy < 0] = -1  # Set the negative indexes

    xy_full_flat = torch.flatten(flat_coo)

    return xy_full_flat
