import numpy as np


def normalize(v):
    """Vector normalization """
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm


def vrrotvec(a, b):
    """Function to rotate one vector to another. Inspired by
    vrrotvec.m in MATLAB.
    Returns rotation axis and rotation angle """
    a = normalize(a)
    b = normalize(b)
    ax = normalize(np.cross(a, b))

    angle = np.arccos(np.minimum(np.dot(a, b), [1]))
    if not np.any(ax):
        absa = np.abs(a)
        mind = np.argmin(absa)
        c = np.zeros((1, 3))
        c[mind] = 0
        ax = normalize(np.cross(a, c))
    r = np.concatenate((ax, angle))
    return r


def vrrotvec2mat(r):
    """Convert the axis-angle representation to the matrix representation of the
    rotation.
    Returns transformation matrix."""
    s = np.sin(r[3])
    c = np.cos(r[3])
    t = 1 - c

    n = normalize(r[0:3])

    x = n[0]
    y = n[1]
    z = n[2]

    m = np.array(
        [[t * x * x + c, t * x * y - s * z, t * x * z + s * y],
         [t * x * y + s * z, t * y * y + c, t * y * z - s * x],
         [t * x * z - s * y, t * y * z + s * x, t * z * z + c]]
    )
    return m


def align_by_axis(sorted_vectors, *axes):
    """
    Parameters
    ----------
    sorted_vectors
    axes

    Returns
    -------

    """

    # Generate names of axes
    bases = {"x": [1, 0, 0], "y": [0, 1, 0], "z": [0, 0, 1]}
    tmp = dict()
    for key, ax in bases.items():
        tmp["-" + key] = [i * -1 for i in ax]
    bases = {**bases, **tmp}

    # Align the first principal axes
    rot1 = vrrotvec(np.array(bases[axes[0]]), sorted_vectors[:, 0])  # return rotation axis and rotation angle
    rmat1 = vrrotvec2mat(rot1)                # return rotation matrix (from principal axis to [1,0,0])
    pa1 = np.matmul(rmat1.T, sorted_vectors)  # then, apply the rotation

    # Align the second principal axes
    rot2 = vrrotvec(np.array(bases[axes[1]]), pa1[:, 1])
    rmat2 = vrrotvec2mat(rot2)
    pa2 = np.matmul(rmat2.T, rmat1.T)

    # Align the third principal axes
    rot3 = vrrotvec(np.array(bases[axes[2]]), pa2[:, 2])
    rmat3 = vrrotvec2mat(rot3)

    # We get the total transformation matrix
    transformation_matrix = np.matmul(rmat3.T, rmat2.T)

    return transformation_matrix


def align(sorted_vectors, proj_direction):
    transformation_matrix = None
    if proj_direction == 1:
        transformation_matrix = xoy_positive_proj(sorted_vectors)
    elif proj_direction == 2:
        transformation_matrix = xoy_negative_proj(sorted_vectors)
    elif proj_direction == 3:
        transformation_matrix = yoz_positive_proj(sorted_vectors)
    elif proj_direction == 4:
        transformation_matrix = yoz_negative_proj(sorted_vectors)
    elif proj_direction == 5:
        transformation_matrix = zox_positive_proj(sorted_vectors)
    elif proj_direction == 6:
        transformation_matrix = zox_negative_proj(sorted_vectors)
    return transformation_matrix


def xoy_positive_proj(sorted_vectors):
    # Align the first principal axes to the X-axes      #-------------------------------------------
    rx = vrrotvec(np.array([1, 0, 0]), sorted_vectors[:, 0])  # return rotation axis and rotation angle
    mx = vrrotvec2mat(rx)                           # return rotation matrix (from principal axis to [1,0,0])
    pa1 = np.matmul(mx.T, sorted_vectors)           # then, apply the rotation

    # Align the second principal axes to the Y-axes
    ry = vrrotvec(np.array([0, 1, 0]), pa1[:, 1])   # As principal axes are orthogonal and (1,0,0) (0,1,0)are orthogonal
    my = vrrotvec2mat(ry)                           # we can rotate to get (0,1,0) result directly
    transformation_matrix = np.matmul(my.T, mx.T)   # We get the total transformation matrix

    return transformation_matrix


def xoy_negative_proj(sorted_vectors):
    # Align the first principal axes to the X-axes      #-------------------------------------------
    rx = vrrotvec(np.array([1, 0, 0]), sorted_vectors[:, 0])  # return rotation axis and rotation angle
    mx = vrrotvec2mat(rx)                           # return rotation matrix (from principal axis to [1,0,0])
    pa1 = np.matmul(mx.T, sorted_vectors)           # then, apply the rotation

    # Align the second principal axes to the Y-axes
    ry = vrrotvec(np.array([0, 1, 0]), pa1[:, 1])   # As principal axes are orthogonal and (1,0,0) (0,1,0)are orthogonal
    my = vrrotvec2mat(ry)
    pa2 = np.matmul(my.T, mx.T)
    # Align the second principal axes to the Z-negative-direction
    rz = vrrotvec(np.array([0, 0, -1]), pa2[:, 2])  # As principal axes are orthogonal and (1,0,0) (0,1,0)are orthogonal
    mz = vrrotvec2mat(rz)
    # we can rotate to get (0,1,0) result directly
    transformation_matrix = np.matmul(mz.T, my.T)   # We get the total transformation matrix

    return transformation_matrix


def yoz_positive_proj(sorted_vectors):
    # Align the first principal axes to the y-axes      #-------------------------------------------
    rx = vrrotvec(np.array([0, 1, 0]), sorted_vectors[:, 0])  # return rotation axis and rotation angle
    mx = vrrotvec2mat(rx)                           # return rotation matrix (from principal axis to [0,1,0])
    pa1 = np.matmul(mx.T, sorted_vectors)           # then, apply the rotation

    # Align the second principal axes to the z-axes
    ry = vrrotvec(np.array([0, 0, 1]), pa1[:, 1])   # As principal axes are orthogonal and (0,1,0)(0,0,1)are orthogonal
    my = vrrotvec2mat(ry)                           # we can rotate to get (0,0,1) result directly
    transformation_matrix = np.matmul(my.T, mx.T)   # We get the total transformation matrix

    return transformation_matrix


def yoz_negative_proj(sorted_vectors):
    # Align the first principal axes to the y-axes      #-------------------------------------------
    rx = vrrotvec(np.array([0, 1, 0]), sorted_vectors[:, 0])  # return rotation axis and rotation angle
    mx = vrrotvec2mat(rx)                           # return rotation matrix (from principal axis to [0,1,0])
    pa1 = np.matmul(mx.T, sorted_vectors)           # then, apply the rotation

    # Align the second principal axes to the z-axes
    ry = vrrotvec(np.array([0, 0, 1]), pa1[:, 1])   # As principal axes are orthogonal and (0,1,0)(0,0,1)are orthogonal
    my = vrrotvec2mat(ry)                           # we can rotate to get (0,0,1) result directly
    pa2 = np.matmul(my.T, mx.T)

    # Align the second principal axes to the Z-negative-direction
    rz = vrrotvec(np.array([-1, 0, 0]), pa2[:, 2])  # As principal axes are orthogonal and (1,0,0) (0,1,0)are orthogonal
    mz = vrrotvec2mat(rz)
    # we can rotate to get (0,1,0) result directly
    transformation_matrix = np.matmul(mz.T, my.T)  # We get the total transformation matrix

    return transformation_matrix


def zox_positive_proj(sorted_vectors):
    # Align the first principal axes to the z-axes      #-------------------------------------------
    rx = vrrotvec(np.array([0, 0, 1]), sorted_vectors[:, 0])  # return rotation axis and rotation angle
    mx = vrrotvec2mat(rx)                           # return rotation matrix (from principal axis to [0,1,0])
    pa1 = np.matmul(mx.T, sorted_vectors)           # then, apply the rotation

    # Align the second principal axes to the x-axes
    ry = vrrotvec(np.array([1, 0, 0]), pa1[:, 1])   # As principal axes are orthogonal and (0,1,0)(0,0,1)are orthogonal
    my = vrrotvec2mat(ry)                           # we can rotate to get (0,0,1) result directly
    transformation_matrix = np.matmul(my.T, mx.T)   # We get the total transformation matrix

    return transformation_matrix


def zox_negative_proj(sorted_vectors):
    # Align the first principal axes to the z-axes      #-------------------------------------------
    rx = vrrotvec(np.array([0, 0, 1]), sorted_vectors[:, 0])  # return rotation axis and rotation angle
    mx = vrrotvec2mat(rx)  # return rotation matrix (from principal axis to [0,1,0])
    pa1 = np.matmul(mx.T, sorted_vectors)  # then, apply the rotation

    # Align the second principal axes to the x-axes
    ry = vrrotvec(np.array([1, 0, 0]), pa1[:, 1])   # As principal axes are orthogonal and (0,1,0)(0,0,1)are orthogonal
    my = vrrotvec2mat(ry)                           # we can rotate to get (0,0,1) result directly
    pa2 = np.matmul(my.T, mx.T)

    # Align the second principal axes to the Z-negative-direction
    rz = vrrotvec(np.array([0, -1, 0]), pa2[:, 2])  # As principal axes are orthogonal and (1,0,0) (0,1,0)are orthogonal
    mz = vrrotvec2mat(rz)
    # we can rotate to get (0,1,0) result directly
    transformation_matrix = np.matmul(mz.T, my.T)  # We get the total transformation matrix

    return transformation_matrix
