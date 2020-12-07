import tensorflow as tf
import numpy as np
import math
import time

def tf_repeat(x, n_repeats):
    #Repeat X for n_repeats time along 0 axis
    #Return a 1D tensor of total number of elements
    rep = tf.ones(shape=[1, n_repeats], dtype = 'int32')
    x = tf.matmul(tf.reshape(x, (-1,1)), rep)
    return tf.reshape(x, [-1])

def tf_interpolate(voxel, x, y, z, out_size):

    """
    Trilinear interpolation for batch of voxels
    :param voxel: The whole voxel grid
    :param x,y,z: indices of voxel
    :param output_size: output size of voxel
    :return:
    """
    batch_size = tf.shape(voxel)[0]
    height = tf.shape(voxel)[1]
    width = tf.shape(voxel)[2]
    depth = tf.shape(voxel)[3]
    n_channels = tf.shape(voxel)[4]

    x = tf.cast(x, 'float32')
    y = tf.cast(y, 'float32')
    z = tf.cast(z, 'float32')

    out_height = out_size[1]
    out_width = out_size[2]
    out_depth = out_size[3]
    out_channel = out_size[4]

    zero = tf.zeros([], dtype='int32')
    max_y = tf.cast(height - 1, 'int32')
    max_x = tf.cast(width - 1, 'int32')
    max_z = tf.cast(depth - 1, 'int32')

    # do sampling
    x0 = tf.cast(tf.floor(x), 'int32')
    x1 = x0 + 1
    y0 = tf.cast(tf.floor(y), 'int32')
    y1 = y0 + 1
    z0 = tf.cast(tf.floor(z), 'int32')
    z1 = z0 + 1

    x0 = tf.clip_by_value(x0, zero, max_x)
    x1 = tf.clip_by_value(x1, zero, max_x)
    y0 = tf.clip_by_value(y0, zero, max_y)
    y1 = tf.clip_by_value(y1, zero, max_y)
    z0 = tf.clip_by_value(z0, zero, max_z)
    z1 = tf.clip_by_value(z1, zero, max_z)

    #A 1D tensor of base indicies describe First index for each shape/map in the whole batch
    #tf.range(batch_size) * width * height * depth : Element to repeat. Each selement in the list is incremented by width*height*depth amount
    # out_height * out_width * out_depth: n of repeat. Create chunks of out_height*out_width*out_depth length with the same value created by tf.rage(batch_size) *width*height*dept
    base = tf_repeat(tf.range(batch_size) * width * height * depth, out_height * out_width * out_depth)

    #Find the Z element of each index

    base_z0 = base + z0 * width * height
    base_z1 = base + z1 * width * height
    #Find the Y element based on Z
    base_z0_y0 = base_z0 + y0 * width
    base_z0_y1 = base_z0 + y1 * width
    base_z1_y0 = base_z1 + y0 * width
    base_z1_y1 = base_z1 + y1 * width

    # Find the X element based on Y, Z for Z=0
    idx_a = base_z0_y0 + x0
    idx_b = base_z0_y1 + x0
    idx_c = base_z0_y0 + x1
    idx_d = base_z0_y1 + x1
    # Find the X element based on Y,Z for Z =1
    idx_e = base_z1_y0 + x0
    idx_f = base_z1_y1 + x0
    idx_g = base_z1_y0 + x1
    idx_h = base_z1_y1 + x1

    # use indices to lookup pixels in the flat image and restore
    # channels dim
    voxel_flat = tf.reshape(voxel, [-1, n_channels])
    voxel_flat = tf.cast(voxel_flat, 'float32')
    Ia = tf.gather(voxel_flat, idx_a)
    Ib = tf.gather(voxel_flat, idx_b)
    Ic = tf.gather(voxel_flat, idx_c)
    Id = tf.gather(voxel_flat, idx_d)
    Ie = tf.gather(voxel_flat, idx_e)
    If = tf.gather(voxel_flat, idx_f)
    Ig = tf.gather(voxel_flat, idx_g)
    Ih = tf.gather(voxel_flat, idx_h)

    # and finally calculate interpolated values
    x0_f = tf.cast(x0, 'float32')
    x1_f = tf.cast(x1, 'float32')
    y0_f = tf.cast(y0, 'float32')
    y1_f = tf.cast(y1, 'float32')
    z0_f = tf.cast(z0, 'float32')
    z1_f = tf.cast(z1, 'float32')

    #First slice XY along Z where z=0
    wa = tf.expand_dims(((x1_f - x) * (y1_f - y) * (z1_f-z)), 1)
    wb = tf.expand_dims(((x1_f - x) * (y - y0_f) * (z1_f-z)), 1)
    wc = tf.expand_dims(((x - x0_f) * (y1_f - y) * (z1_f-z)), 1)
    wd = tf.expand_dims(((x - x0_f) * (y - y0_f) * (z1_f-z)), 1)
    # First slice XY along Z where z=1
    we = tf.expand_dims(((x1_f - x) * (y1_f - y) * (z-z0_f)), 1)
    wf = tf.expand_dims(((x1_f - x) * (y - y0_f) * (z-z0_f)), 1)
    wg = tf.expand_dims(((x - x0_f) * (y1_f - y) * (z-z0_f)), 1)
    wh = tf.expand_dims(((x - x0_f) * (y - y0_f) * (z-z0_f)), 1)


    output = tf.add_n([wa * Ia, wb * Ib, wc * Ic, wd * Id,  we * Ie, wf * If, wg * Ig, wh * Ih])
    return output

def tf_voxel_meshgrid(height, width, depth, homogeneous = False):
    with tf.variable_scope('voxel_meshgrid'):
        #Because 'ij' ordering is used for meshgrid, z_t and x_t are swapped (Think about order in 'xy' VS 'ij'
        x_t, y_t, z_t = tf.meshgrid(tf.range(depth, dtype = tf.float32),
                                    tf.range(height, dtype = tf.float32),
                                    tf.range(width, dtype = tf.float32), indexing='ij')
        #Reshape into a big list of slices one after another along the X,Y,Z direction
        x_t_flat = tf.reshape(x_t, (1, -1))
        y_t_flat = tf.reshape(y_t, (1, -1))
        z_t_flat = tf.reshape(z_t, (1, -1))

        #Vertical stack to create a (3,N) matrix for X,Y,Z coordinates
        grid = tf.concat([x_t_flat, y_t_flat, z_t_flat], axis=0)
        if homogeneous:
            ones = tf.ones_like(x_t_flat)
            grid = tf.concat([grid, ones], axis = 0)
        return grid

def tf_rotation_around_grid_centroid(view_params, shapenet_viewer = False):
    """
    :param view_params: batch of view parameters. Shape : [batch_size, 2]
    :param radius:
    :param useX: USe when X axis and Z axis are switched
    :return:
    """
    #This function returns a rotation matrix around a center with y-axis being the up vector.
    #It first rotates the matrix by the azimuth angle (theta) around y, then around X-axis by elevation angle (gamma)
    #return a rotation matrix in homogenous coordinate
    #The default Open GL camera is to looking towards the negative Z direction
    #This function is suitable when the silhoutte projection is done along the Z direction
    batch_size = tf.shape(view_params)[0]

    azimuth    = tf.reshape(view_params[:, 0], (batch_size, 1, 1))
    elevation  = tf.reshape(view_params[:, 1], (batch_size, 1, 1))

    # azimuth = azimuth
    if shapenet_viewer == False:
        azimuth = (azimuth - tf.constant(math.pi * 0.5))

    #========================================================
    #Because tensorflow does not allow tensor item replacement
    #A new matrix needs to be created from scratch by concatenating different vectors into rows and stacking them up
    #Batch Rotation Y matrixes
    ones = tf.ones_like(azimuth)
    zeros = tf.zeros_like(azimuth)
    batch_Rot_Y = tf.concat([
        tf.concat([tf.cos(azimuth),  zeros, -tf.sin(azimuth), zeros], axis=2),
        tf.concat([zeros, ones,  zeros,zeros], axis=2),
        tf.concat([tf.sin(azimuth),  zeros, tf.cos(azimuth), zeros], axis=2),
        tf.concat([zeros, zeros, zeros, ones], axis=2)], axis=1)

    # #Batch Rotation Z matrixes
    # batch_Rot_Z = tf.concat([
    #     tf.concat([tf.cos(elevation),  tf.sin(elevation),  zeros, zeros], axis=2),
    #     tf.concat([-tf.sin(elevation), tf.cos(elevation),  zeros, zeros], axis=2),
    #     tf.concat([zeros, zeros, ones,  zeros], axis=2),
    #     tf.concat([zeros, zeros, zeros, ones], axis=2)], axis=1)

    batch_Rot_X = tf.concat([
        tf.concat([ones,  zeros,  zeros, zeros], axis=2),
        tf.concat([zeros, tf.cos(elevation),  -tf.sin(elevation), zeros], axis=2),
        tf.concat([zeros, tf.sin(elevation),  tf.cos(elevation),  zeros], axis=2),
        tf.concat([zeros, zeros,  zeros, ones], axis=2)], axis=1)

    transformation_matrix = tf.matmul(batch_Rot_X, batch_Rot_Y)
    if tf.shape(view_params)[1] == 2:
        return transformation_matrix
    else:
    #Batch Scale matrixes:
        scale = tf.reshape(view_params[:, 2], (batch_size, 1, 1))
        batch_Scale= tf.concat([
            tf.concat([scale,  zeros,  zeros, zeros], axis=2),
            tf.concat([zeros, scale,  zeros, zeros], axis=2),
            tf.concat([zeros, zeros,  scale,  zeros], axis=2),
            tf.concat([zeros, zeros,  zeros, ones], axis=2)], axis=1)
    return transformation_matrix, batch_Scale

def tf_rotation_resampling(voxel_array, transformation_matrix, params, Scale_matrix = None, size=64, new_size=128):
    """
    Batch resampling function
    :param voxel_array: batch of voxels. Shape = [batch_size, height, width, depth, features]
    :param transformation_matrix: Rotation matrix. Shape = [batch_size, height, width, depth, features]
    :param size:
    :param new_size:
    :return:
    """

    batch_size = tf.shape(voxel_array)[0]
    n_channels = voxel_array.get_shape()[4].value
    target = tf.zeros([ batch_size, new_size, new_size, new_size])
    #Aligning the centroid of the object (voxel grid) to origin for rotation,
    #then move the centroid back to the original position of the grid centroid
    T = tf.constant([[1,0,0, -size * 0.5],
                  [0,1,0, -size * 0.5],
                  [0,0,1, -size * 0.5],
                  [0,0,0,1]])
    T = tf.tile(tf.reshape(T, (1, 4, 4)), [batch_size, 1, 1])

    # However, since the rotated grid might be out of bound for the original grid size,
    # move the rotated grid to a new bigger grid
    T_new_inv = tf.constant([[1, 0, 0, new_size * 0.5],
                             [0, 1, 0, new_size * 0.5],
                             [0, 0, 1, new_size * 0.5],
                             [0, 0, 0, 1]])
    T_new_inv = tf.tile(tf.reshape(T_new_inv, (1, 4, 4)), [batch_size, 1, 1])


    # Add the actual shifting in x and y dimension accoding to input param
    x_shift = tf.reshape(params[:, 3], (batch_size, 1, 1))
    y_shift = tf.reshape(params[:, 4], (batch_size, 1, 1))
    z_shift = tf.reshape(params[:, 5], (batch_size, 1, 1))
    # ========================================================
    # Because tensorflow does not allow tensor item replacement
    # A new matrix needs to be created from scratch by concatenating different vectors into rows and stacking them up
    # Batch Rotation Y matrixes
    ones = tf.ones_like(x_shift)
    zeros = tf.zeros_like(x_shift)

    T_translate = tf.concat([
        tf.concat([ones, zeros, zeros, x_shift], axis=2),
        tf.concat([zeros, ones, zeros, y_shift], axis=2),
        tf.concat([zeros, zeros, ones, z_shift], axis=2),
        tf.concat([zeros, zeros, zeros, ones], axis=2)], axis=1)
    total_M = tf.matmul(tf.matmul(tf.matmul(tf.matmul(T_new_inv, T_translate), Scale_matrix), transformation_matrix), T)


    try:
        total_M = tf.matrix_inverse(total_M)

        total_M = total_M[:, 0:3, :] #Ignore the homogenous coordinate so the results are 3D vectors
        grid = tf_voxel_meshgrid(new_size, new_size, new_size, homogeneous=True)
        grid = tf.tile(tf.reshape(grid, (1, tf.to_int32(grid.get_shape()[0]) , tf.to_int32(grid.get_shape()[1]))), [batch_size, 1, 1])
        grid_transform = tf.matmul(total_M, grid)
        x_s_flat = tf.reshape(grid_transform[:, 0, :], [-1])
        y_s_flat = tf.reshape(grid_transform[:, 1, :], [-1])
        z_s_flat = tf.reshape(grid_transform[:, 2, :], [-1])
        input_transformed = tf_interpolate(voxel_array, x_s_flat, y_s_flat, z_s_flat,[batch_size, new_size, new_size, new_size, n_channels])
        target= tf.reshape(input_transformed, [batch_size, new_size, new_size, new_size, n_channels])

        return target, grid_transform
    except tf.InvalidArgumentError:
        return None

def tf_rotation_resampling_skew(voxel_array, transformation_matrix, skew_matrix, params, Scale_matrix = None, size=64, new_size=128):
    """
    Batch resampling function
    :param voxel_array: batch of voxels. Shape = [batch_size, height, width, depth, features]
    :param transformation_matrix: Rotation matrix. Shape = [batch_size, height, width, depth, features]
    :param size:
    :param new_size:
    :return:
    """

    batch_size = tf.shape(voxel_array)[0]
    n_channels = voxel_array.get_shape()[4].value
    target = tf.zeros([ batch_size, new_size, new_size, new_size])
    #Aligning the centroid of the object (voxel grid) to origin for rotation,
    #then move the centroid back to the original position of the grid centroid
    T = tf.constant([[1,0,0, -size * 0.5],
                  [0,1,0, -size * 0.5],
                  [0,0,1, -size * 0.5],
                  [0,0,0,1]])
    T = tf.tile(tf.reshape(T, (1, 4, 4)), [batch_size, 1, 1])

    # However, since the rotated grid might be out of bound for the original grid size,
    # move the rotated grid to a new bigger grid
    T_new_inv = tf.constant([[1, 0, 0, new_size * 0.5],
                             [0, 1, 0, new_size * 0.5],
                             [0, 0, 1, new_size * 0.5],
                             [0, 0, 0, 1]])
    T_new_inv = tf.tile(tf.reshape(T_new_inv, (1, 4, 4)), [batch_size, 1, 1])


    # Add the actual shifting in x and y dimension accoding to input param
    x_shift = tf.reshape(params[:, 3], (batch_size, 1, 1))
    y_shift = tf.reshape(params[:, 4], (batch_size, 1, 1))
    z_shift = tf.reshape(params[:, 5], (batch_size, 1, 1))
    # ========================================================
    # Because tensorflow does not allow tensor item replacement
    # A new matrix needs to be created from scratch by concatenating different vectors into rows and stacking them up
    # Batch Rotation Y matrixes
    ones = tf.ones_like(x_shift)
    zeros = tf.zeros_like(x_shift)

    T_translate = tf.concat([
        tf.concat([ones, zeros, zeros, x_shift], axis=2),
        tf.concat([zeros, ones, zeros, y_shift], axis=2),
        tf.concat([zeros, zeros, ones, z_shift], axis=2),
        tf.concat([zeros, zeros, zeros, ones], axis=2)], axis=1)



    #===========================================================
    try:
        skew_inv = tf.matrix_inverse(skew_matrix)
    except:
        with tf.device('/cpu:0'):
            skew_inv = tf.matrix_inverse(skew_matrix)
    total_M = tf.matmul(tf.matmul(tf.matmul(skew_inv, T_translate), Scale_matrix), transformation_matrix)   # projection + rotation
    total_M = tf.matmul(tf.matmul(T_new_inv, total_M), T)


    total_M = tf.matrix_inverse(total_M)

    # total_M = total_M[:, 0:3, :] #Ignore the homogenous coordinate so the results are 3D vectors
    grid = tf_voxel_meshgrid(new_size, new_size, new_size, homogeneous=True)
    grid = tf.tile(tf.reshape(grid, (1, tf.to_int32(grid.get_shape()[0]) , tf.to_int32(grid.get_shape()[1]))), [batch_size, 1, 1])
    grid_transform = tf.matmul(total_M, grid)

    # division by homogeneous coordinate
    homo_coor = tf.expand_dims(grid_transform[:, 3, :], 1)
    homo_coor = tf.tile(homo_coor, (1, 4, 1))
    grid_transform = tf.div(grid_transform, homo_coor)

    x_s_flat = tf.reshape(grid_transform[:, 0, :], [-1])
    y_s_flat = tf.reshape(grid_transform[:, 1, :], [-1])
    z_s_flat = tf.reshape(grid_transform[:, 2, :], [-1])
    input_transformed = tf_interpolate(voxel_array, x_s_flat, y_s_flat, z_s_flat,[batch_size, new_size, new_size, new_size, n_channels])
    target= tf.reshape(input_transformed, [batch_size, new_size, new_size, new_size, n_channels])

    return target, grid_transform

def tf_3D_transform(voxel_array, view_params, size=64, new_size=128, shapenet_viewer=False):
    M, S = tf_rotation_around_grid_centroid(view_params[:, :3], shapenet_viewer=shapenet_viewer)
    target, grids = tf_rotation_resampling(voxel_array, M, params=view_params, Scale_matrix=S, size = size, new_size=new_size)
    return target

def tf_3D_transform_skew(voxel_array, view_params, skew_matrix, size=64, new_size=128, shapenet_viewer=False):
    M, S = tf_rotation_around_grid_centroid(view_params[:, :3], shapenet_viewer=shapenet_viewer)
    target, grids = tf_rotation_resampling_skew(voxel_array, M, skew_matrix, params=view_params, Scale_matrix=S, size = size, new_size=new_size)
    return target

def generate_random_rotation_translation(batch_size, elevation_low=10, elevation_high=170, azimuth_low=0, azimuth_high=359,
                                         scale_low=1.0, scale_high=1.0,
                                         transX_low=-3, transX_high=3,
                                         transY_low=-3, transY_high=3,
                                         transZ_low=-3, transZ_high=3,
                                         with_translation=False, with_scale=False):
    params = np.zeros((batch_size, 6))
    column = np.arange(0, batch_size)
    azimuth = np.random.randint(azimuth_low, azimuth_high, (batch_size)).astype(np.float) * math.pi / 180.0
    temp = np.random.randint(elevation_low, elevation_high, (batch_size))
    elevation = (90. - temp.astype(np.float)) * math.pi / 180.0
    params[column, 0] = azimuth
    params[column, 1] = elevation

    if with_translation:
        shift_x = transX_low + np.random.random(batch_size) * (transX_high - transX_low)
        shift_y = transY_low + np.random.random(batch_size) * (transY_high - transY_low)
        shift_z = transZ_low + np.random.random(batch_size) * (transZ_high - transZ_low)
        params[column, 3] = shift_x
        params[column, 4] = shift_y
        params[column, 5] = shift_z

    if with_scale:
        scale = float(np.random.uniform(scale_low, scale_high))
        params[column, 2] = scale
    else:
        params[column, 2] = 1.0

    return params


def generate_random_rotation_translation_2objs(batch_size, elevation_low=90, elevation_high=91, azimuth_low=0, azimuth_high=359, scale_low=0.8, scale_high=1.5,
                                         transX_low=-3, transX_high=3,  transY_low=-3, transY_high=3, transZ_low=-3, transZ_high=3,
                                         with_translation=False, with_scale=False, margin=None):
   #Sampling translation in as integers, not as floats

    params = np.zeros((batch_size, 12))
    for i in range(batch_size):
        azimuth = np.random.randint(azimuth_low, azimuth_high, 2).astype(np.float) * math.pi / 180.0
        temp = np.random.randint(elevation_low, elevation_high, 2)
        elevation = (90. - temp.astype(np.float)) * math.pi / 180.0
        params[i, 0] = azimuth[0]
        params[i, 1] = elevation[0]
        params[i, 6] = azimuth[1]
        params[i, 7] = elevation[1]


        if with_translation:
            if transX_low==0 and transX_high==0:
                shift_x = [0, 0]
            else:
                shift_x = np.random.randint(transX_low, transX_high, 2)

            if transY_low == 0 and transY_high==0:
                shift_y = [0, 0]
            else:
                shift_y = np.random.randint(transY_low, transY_high, 2)


            shift_z = np.random.randint(transZ_low + 1 , transZ_high)

            # shift_x2 = np.random.randint(transX_low, transX_high)
            # shift_y2 = np.random.randint(transY_low, transY_high)
            try:
                shift_z2 = np.random.randint(transZ_low, shift_z)
            except:
                print(shift_z)

            params[i, 3] = shift_x[0]
            params[i, 4] = shift_y[0]
            params[i, 5] = shift_z
            params[i, 9]  = shift_x[1]
            params[i, 10] = shift_y[1]
            params[i, 11] = shift_z2
        else:
            shift_x = 0
            shift_y = 0
            shift_z = 0
            params[i, 3] = shift_x
            params[i, 4] = shift_y
            params[i, 5] = shift_z

        if with_scale:
            scale = np.random.choice([scale_low, scale_high], 2).astype(np.float)
            params[i, 2] = scale[0]
            params[i, 8] = scale[1]
        else:
            params[i, 2] = 1.0
            params[i, 8] = 1.0
    return params

def compute_skew_matrix_nearFar(batch_size, size, new_size, focal_length=35, sensor_size=32, distance = 10):
    ## Corners of the "normalized device coordinates" (i.e. output grid)
    x1 = x4 = -new_size / 2 + 0.5
    x2 = x3 = new_size / 2 - 0.5
    y1 = y2 = new_size / 2 - 0.5
    y3 = y4 = -new_size / 2 + 0.5

    ## Corners of the frustum (take 3: introducing z near/far).
    #Hardcoded d for debug
    # d = np.sqrt(3) * size  # distance of camera from origin [unit: input voxels]
    # d = 0.75 * size

    z_near = 0.75 * distance  # a quarter of the way from origin towards camera [input voxels]
    z_far = 1.25 * distance  # a quarter of the way from origin away from camera [input voxels]

    # d = 20; z_near = d - 2; z_far = d + 2  # fixed test

    # print(f"d = {d} || z_near ... z_far = {z_near} ... {z_far}")
    x1p = -sensor_size * z_far / (2 * focal_length)
    x2p = sensor_size * z_far / (2 * focal_length)
    x3p = sensor_size * z_near / (2 * focal_length)
    x4p = -sensor_size * z_near / (2 * focal_length)
    y1p = distance - z_near;
    y2p = distance - z_near;
    y3p = distance - z_far;
    y4p = distance - z_far
    # print(f"(x1p, y2p) = ({x1p}, {y1p})")

    ## Solve homography from 4 point correspondences (general case)
    ## Source: https://math.stackexchange.com/a/2619023
    PMat = np.mat([
        [-x1, -y1, -1, 0, 0, 0, x1 * x1p, y1 * x1p, x1p],
        [0, 0, 0, -x1, -y1, -1, x1 * y1p, y1 * y1p, y1p],
        [-x2, -y2, -1, 0, 0, 0, x2 * x2p, y2 * x2p, x2p],
        [0, 0, 0, -x2, -y2, -1, x2 * y2p, y2 * y2p, y2p],
        [-x3, -y3, -1, 0, 0, 0, x3 * x3p, y3 * x3p, x3p],
        [0, 0, 0, -x3, -y3, -1, x3 * y3p, y3 * y3p, y3p],
        [-x4, -y4, -1, 0, 0, 0, x4 * x4p, y4 * x4p, x4p],
        [0, 0, 0, -x4, -y4, -1, x4 * y4p, y4 * y4p, y4p],
        [0, 0, 0, 0, 0, 0, 0, 0, 1]
    ])
    H = PMat.I * np.mat([[0, 0, 0, 0, 0, 0, 0, 0, 1]]).T
    H = H.reshape([3, 3])
    H_3D = np.mat([[H[0, 0], 0, 0, 0],
                   [0, H[0, 0], 0, 0],
                   [0, 0, H[1, 1], H[1, 2]],
                   [0, 0, H[2, 1], 1.]])

    skew_3D_batch = np.tile(np.expand_dims(H_3D, 0), [batch_size, 1 ,1])
    return skew_3D_batch


