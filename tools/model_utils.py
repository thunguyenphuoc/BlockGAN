import numpy as np
import tensorflow as tf

import math
import os
import glob
import scipy.io


#=======================================================================================================================
#Helper functions to load pretrained weights
#=======================================================================================================================

def get_weight(weight_name, weight_dict):
    if weight_dict is None:
        print("Can't find weight")
        return None
    else:
        return weight_dict.get(weight_name)  # returns None if name is not found in dictionary

def load_weights(weight_dir):
    weight_path_all = glob.glob(os.path.join(weight_dir, "*.txt.npz"))
    pretrained_weight_dict = {}
    print(len(weight_path_all))
    for path in weight_path_all:
        with np.load(path) as data:
            layer_name = os.path.basename(path).split('.')[0]
            print(layer_name)
            pretrained_weight_dict[layer_name] = data['arr_0']
            print(data['arr_0'].shape)
    return pretrained_weight_dict

def load_z_mapping_function(z, output_channel, weight, bias, scope, act=None):
    with tf.variable_scope(scope) as sc:
        w = tf.get_variable('w', initializer=weight, trainable=False)
        b = tf.get_variable('biases', initializer=bias, trainable=False)
        if act == "lrelu":
            print ("LRELU")
            out = lrelu(tf.matmul(z, w) + b)
        else:
          out = act(tf.matmul(z, w) + b)
        return out[:, :output_channel], out[:, output_channel:]

def load_weights(weight_dir):
    weight_path_all = glob.glob(os.path.join(weight_dir, "*.txt.npz"))
    pretrained_weight_dict = {}
    print(len(weight_path_all))
    for path in weight_path_all:
        with np.load(path) as data:
            layer_name = os.path.basename(path).split('.')[0]
            print(layer_name)
            pretrained_weight_dict[layer_name] = data['arr_0']
    return pretrained_weight_dict

#=======================================================================================================================
def save_txt_file(pred, name, SAVE_DIR):
    with open(os.path.join(SAVE_DIR, "{0}.txt".format(name)), 'w') as fp:
        for i in pred:
            # print(tuple(point.tolist()))
            fp.write("{0}\n".format(i))

def transform_tensor_to_image (tensor):
    t = tf.transpose(tensor, [0 , 2, 1, 3])
    return t[:,::-1, :, :]

def transform_voxel_to_match_image(tensor):
    tensor = tf.transpose(tensor, [0, 2, 1, 3, 4])
    tensor = tensor[:, ::-1, :, :, :]
    return tensor

def transform_image_to_match_voxel(tensor):
    tensor = tf.transpose(tensor, [0, 2, 1, 3])
    tensor = tensor[:, ::-1, :, :]
    return tensor

def np_transform_tensor_to_image (tensor):
    t = np.transpose(tensor, [0, 2, 1, 3])
    return t

