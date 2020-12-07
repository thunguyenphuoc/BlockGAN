from __future__ import division
import os
import sys
from glob import glob
import json
import shutil


with open(sys.argv[1], 'r') as fh:
    cfg=json.load(fh)
IMAGE_PATH       = cfg['image_path']
OUTPUT_DIR = cfg['output_dir']
LOGDIR = os.path.join(OUTPUT_DIR, "log")


from tools.ops import *
from tools.utils import get_image, merge, inverse_transform, to_bool
from tools.rotation_utils import *
from tools.model_utils import transform_voxel_to_match_image

class BlockGAN(object):
  def __init__(self, sess, input_height=108, input_width=108, crop=True,
         output_height=64, output_width=64,
         gf_dim=64, df_dim=64,
         c_dim=3, dataset_name='lsun',
         input_fname_pattern='*.webp'):
    """
    Args:
      sess: TensorFlow session
      batch_size: The size of batch. Should be specified before training.
      y_dim: (optional) Dimension of dim for y. [None]
      z_dim: (optional) Dimension of dim for Z. [100]
      gf_dim: (optional) Dimension of gen filters in first conv layer. [64]
      df_dim: (optional) Dimension of discrim filters in first conv layer. [64]
      gfc_dim: (optional) Dimension of gen units for for fully connected layer. [1024]
      dfc_dim: (optional) Dimension of discrim units for fully connected layer. [1024]
      c_dim: (optional) Dimension of image color. For grayscale input, set to 1. [3]
    """
    self.sess = sess
    self.crop = crop

    self.input_height = input_height
    self.input_width = input_width
    self.output_height = output_height
    self.output_width = output_width

    self.gf_dim = gf_dim
    self.df_dim = df_dim
    self.c_dim = c_dim

    self.dataset_name = dataset_name
    self.input_fname_pattern = input_fname_pattern
    self.data = glob.glob(os.path.join(IMAGE_PATH, self.input_fname_pattern))

    self.checkpoint_dir = LOGDIR

  def build(self, build_func_name):
      build_func = eval("self." + build_func_name)
      build_func()

  def build_loss(self):
      if str(cfg['loss_func']).lower() == 'lsgan':
          print("LSGAN")
          self.d_loss_real = 0.5 * tf.reduce_mean(tf.squared_difference(self.D_logits, 1.0))
          self.d_loss_fake = 0.5 * tf.reduce_mean(tf.square(self.D_logits_))
          self.d_loss = self.d_loss_real + self.d_loss_fake
          self.g_loss = 0.5 * tf.reduce_mean(tf.squared_difference(self.D_logits_, 1.0))
      elif str(cfg['loss_func']).lower() == 'hinge':
          print('HingeGAN')
          self.d_loss_real = tf.reduce_mean(tf.nn.relu(1.0 - self.D_logits))
          self.d_loss_fake = tf.reduce_mean(tf.nn.relu(1.0 + self.D_logits_))
          self.d_loss = self.d_loss_real + self.d_loss_fake
          self.g_loss = -tf.reduce_mean(self.D_logits_)
      elif str(cfg['loss_func']).lower() == 'gan':
          print("Normal GAN")
          self.d_loss_real = tf.reduce_mean(sigmoid_cross_entropy_with_logits(self.D_logits, tf.ones_like(self.D)))
          self.d_loss_fake = tf.reduce_mean(sigmoid_cross_entropy_with_logits(self.D_logits_, tf.zeros_like(self.D_)))
          self.d_loss = self.d_loss_real + self.d_loss_fake
          self.g_loss = tf.reduce_mean(sigmoid_cross_entropy_with_logits(self.D_logits_, tf.ones_like(self.D_)))

  def build_BlockGAN_multi(self):
    self.view_in = tf.placeholder(tf.float32, [None, 6], name='view_in')
    self.view_in2 = tf.placeholder(tf.float32, [None, 6], name='view_in2')
    self.skew_matrix_in = tf.placeholder(tf.float32, [None, 4, 4], name="skew_matrix_in")
    self.inputs = tf.placeholder(tf.float32, [None, self.output_height, self.output_width, self.c_dim], name='real_images')
    self.z = tf.placeholder(tf.float32, [None, cfg['z_dim']], name='z')
    self.z2 = tf.placeholder(tf.float32, [None, cfg['z_dim2']], name='z2')
    inputs = self.inputs

    gen_func = eval("self." + (cfg['generator']))
    dis_func = eval("self." + (cfg['discriminator']))

    self.skew_func = eval(cfg['skew_func'])
    self.gen_view_func = eval(cfg['view_func'])
    self.G = gen_func(self.z, self.z2, self.view_in, self.view_in2, self.skew_matrix_in)


    if str.lower(str(cfg["style_disc"])) == "true":
        if self.output_height == 128:
            print("Style Disc")
            self.D, self.D_logits, _, self.d_h1_r, self.d_h2_r, self.d_h3_r, self.d_h4_r = dis_func(inputs, cont_dim=cfg['z_dim'],
                                                                                       reuse=False)
            self.D_, self.D_logits_, self.Q_c_given_x, self.d_h1_f, self.d_h2_f, self.d_h3_f, self.d_h4_f = dis_func(self.G,
                                                                                                        cont_dim=cfg[
                                                                                                            'z_dim'],
                                                                                                        reuse=True)

            self.d_h1_loss = cfg["DStyle_lambda"] * (
                    tf.reduce_mean(sigmoid_cross_entropy_with_logits(self.d_h1_r, tf.ones_like(self.d_h1_r))) \
                    + tf.reduce_mean(sigmoid_cross_entropy_with_logits(self.d_h1_f, tf.zeros_like(self.d_h1_f))))
            self.d_h2_loss = cfg["DStyle_lambda"] * (
                    tf.reduce_mean(sigmoid_cross_entropy_with_logits(self.d_h2_r, tf.ones_like(self.d_h2_r))) \
                    + tf.reduce_mean(sigmoid_cross_entropy_with_logits(self.d_h2_f, tf.zeros_like(self.d_h2_f))))
            self.d_h3_loss = cfg["DStyle_lambda"] * (
                    tf.reduce_mean(sigmoid_cross_entropy_with_logits(self.d_h3_r, tf.ones_like(self.d_h3_r))) \
                    + tf.reduce_mean(sigmoid_cross_entropy_with_logits(self.d_h3_f, tf.zeros_like(self.d_h3_f))))
            self.d_h4_loss = cfg["DStyle_lambda"] * (
                        tf.reduce_mean(sigmoid_cross_entropy_with_logits(self.d_h4_r, tf.ones_like(self.d_h4_r))) \
                        + tf.reduce_mean(sigmoid_cross_entropy_with_logits(self.d_h4_f, tf.zeros_like(self.d_h4_f))))
            self.d_style_loss = self.d_h1_loss + self.d_h2_loss + self.d_h3_loss + self.d_h4_loss
        else:
            print("Style Disc")
            self.D, self.D_logits, _, self.d_h1_r, self.d_h2_r, self.d_h3_r = dis_func(inputs, cont_dim=cfg['z_dim'], reuse=False)
            self.D_, self.D_logits_, self.Q_c_given_x, self.d_h1_f, self.d_h2_f, self.d_h3_f = dis_func(self.G, cont_dim=cfg['z_dim'], reuse=True)

            self.d_h1_loss = cfg["DStyle_lambda"] * (
                        tf.reduce_mean(sigmoid_cross_entropy_with_logits(self.d_h1_r, tf.ones_like(self.d_h1_r))) \
                        + tf.reduce_mean(sigmoid_cross_entropy_with_logits(self.d_h1_f, tf.zeros_like(self.d_h1_f))))
            self.d_h2_loss = cfg["DStyle_lambda"] * (
                        tf.reduce_mean(sigmoid_cross_entropy_with_logits(self.d_h2_r, tf.ones_like(self.d_h2_r))) \
                        + tf.reduce_mean(sigmoid_cross_entropy_with_logits(self.d_h2_f, tf.zeros_like(self.d_h2_f))))
            self.d_h3_loss = cfg["DStyle_lambda"] * (
                        tf.reduce_mean(sigmoid_cross_entropy_with_logits(self.d_h3_r, tf.ones_like(self.d_h3_r))) \
                        + tf.reduce_mean(sigmoid_cross_entropy_with_logits(self.d_h3_f, tf.zeros_like(self.d_h3_f))))
            self.d_style_loss = self.d_h1_loss + self.d_h2_loss + self.d_h3_loss
    else:
        self.D, self.D_logits, _ = dis_func(inputs, cont_dim=cfg['z_dim'], reuse=False)
        self.D_, self.D_logits_, self.Q_c_given_x = dis_func(self.G, cont_dim=cfg['z_dim'], reuse=True)


    self.d_loss_real = tf.reduce_mean(sigmoid_cross_entropy_with_logits(self.D_logits, tf.ones_like(self.D)))
    self.d_loss_fake = tf.reduce_mean(sigmoid_cross_entropy_with_logits(self.D_logits_, tf.zeros_like(self.D_)))
    self.d_loss = self.d_loss_real + self.d_loss_fake
    self.g_loss = tf.reduce_mean(sigmoid_cross_entropy_with_logits(self.D_logits_, tf.ones_like(self.D_)))

    if str.lower(str(cfg["style_disc"])) == "true":
        print("Style disc")
        self.d_loss = self.d_loss + self.d_style_loss


    self.d_loss_real_sum = scalar_summary("d_loss_real", self.d_loss_real)
    self.d_loss_fake_sum = scalar_summary("d_loss_fake", self.d_loss_fake)
    self.g_loss_sum = scalar_summary("g_loss", self.g_loss)
    self.d_loss_sum = scalar_summary("d_loss", self.d_loss)

    t_vars = tf.trainable_variables()

    self.d_vars = [var for var in t_vars if 'd_' in var.name]
    self.g_vars = [var for var in t_vars if 'g_' in var.name]

    self.saver = tf.train.Saver()

  def build_HoloGAN_multi_skew_loss(self):
    self.view_in = tf.placeholder(tf.float32, [None, 6], name='view_in')
    self.view_in2 = tf.placeholder(tf.float32, [None, 6], name='view_in2')
    self.skew_matrix_in = tf.placeholder(tf.float32, [None, 4, 4], name="skew_matrix_in")
    self.inputs = tf.placeholder(tf.float32, [None, self.output_height, self.output_width, self.c_dim], name='real_images')
    self.z = tf.placeholder(tf.float32, [None, cfg['z_dim']], name='z')
    self.z2 = tf.placeholder(tf.float32, [None, cfg['z_dim2']], name='z2')
    inputs = self.inputs

    gen_func = eval("self." + (cfg['generator']))
    dis_func = eval("self." + (cfg['discriminator']))
    self.gen_feat_func = eval("self." + (cfg['generator_feat']))

    self.skew_func = eval(cfg['skew_func'])
    self.gen_view_func = eval(cfg['view_func'])
    self.G = gen_func(self.z, self.z2, self.view_in, self.view_in2, self.skew_matrix_in)


    if str.lower(str(cfg["style_disc"])) == "true":
        if self.output_height == 128:
            print("Style Disc")
            self.D, self.D_logits, _, self.d_h1_r, self.d_h2_r, self.d_h3_r, self.d_h4_r = dis_func(inputs, cont_dim=cfg['z_dim'],
                                                                                       reuse=False)
            self.D_, self.D_logits_, self.Q_c_given_x, self.d_h1_f, self.d_h2_f, self.d_h3_f, self.d_h4_f = dis_func(self.G,
                                                                                                        cont_dim=cfg[
                                                                                                            'z_dim'],
                                                                                                        reuse=True)

            self.d_h1_loss = cfg["DStyle_lambda"] * (
                    tf.reduce_mean(sigmoid_cross_entropy_with_logits(self.d_h1_r, tf.ones_like(self.d_h1_r))) \
                    + tf.reduce_mean(sigmoid_cross_entropy_with_logits(self.d_h1_f, tf.zeros_like(self.d_h1_f))))
            self.d_h2_loss = cfg["DStyle_lambda"] * (
                    tf.reduce_mean(sigmoid_cross_entropy_with_logits(self.d_h2_r, tf.ones_like(self.d_h2_r))) \
                    + tf.reduce_mean(sigmoid_cross_entropy_with_logits(self.d_h2_f, tf.zeros_like(self.d_h2_f))))
            self.d_h3_loss = cfg["DStyle_lambda"] * (
                    tf.reduce_mean(sigmoid_cross_entropy_with_logits(self.d_h3_r, tf.ones_like(self.d_h3_r))) \
                    + tf.reduce_mean(sigmoid_cross_entropy_with_logits(self.d_h3_f, tf.zeros_like(self.d_h3_f))))
            self.d_h4_loss = cfg["DStyle_lambda"] * (
                        tf.reduce_mean(sigmoid_cross_entropy_with_logits(self.d_h4_r, tf.ones_like(self.d_h4_r))) \
                        + tf.reduce_mean(sigmoid_cross_entropy_with_logits(self.d_h4_f, tf.zeros_like(self.d_h4_f))))
            self.d_style_loss = self.d_h1_loss + self.d_h2_loss + self.d_h3_loss + self.d_h4_loss
        else:
            print("Style Disc")
            self.D, self.D_logits, _, self.d_h1_r, self.d_h2_r, self.d_h3_r = dis_func(inputs, cont_dim=cfg['z_dim'], reuse=False)
            self.D_, self.D_logits_, self.Q_c_given_x, self.d_h1_f, self.d_h2_f, self.d_h3_f = dis_func(self.G, cont_dim=cfg['z_dim'], reuse=True)

            self.d_h1_loss = cfg["DStyle_lambda"] * (
                        tf.reduce_mean(sigmoid_cross_entropy_with_logits(self.d_h1_r, tf.ones_like(self.d_h1_r))) \
                        + tf.reduce_mean(sigmoid_cross_entropy_with_logits(self.d_h1_f, tf.zeros_like(self.d_h1_f))))
            self.d_h2_loss = cfg["DStyle_lambda"] * (
                        tf.reduce_mean(sigmoid_cross_entropy_with_logits(self.d_h2_r, tf.ones_like(self.d_h2_r))) \
                        + tf.reduce_mean(sigmoid_cross_entropy_with_logits(self.d_h2_f, tf.zeros_like(self.d_h2_f))))
            self.d_h3_loss = cfg["DStyle_lambda"] * (
                        tf.reduce_mean(sigmoid_cross_entropy_with_logits(self.d_h3_r, tf.ones_like(self.d_h3_r))) \
                        + tf.reduce_mean(sigmoid_cross_entropy_with_logits(self.d_h3_f, tf.zeros_like(self.d_h3_f))))
            self.d_style_loss = self.d_h1_loss + self.d_h2_loss + self.d_h3_loss
    else:
        self.D, self.D_logits, _ = dis_func(inputs, cont_dim=cfg['z_dim'], reuse=False)
        self.D_, self.D_logits_, self.Q_c_given_x = dis_func(self.G, cont_dim=cfg['z_dim'], reuse=True)


    self.build_loss()

    if str.lower(str(cfg["style_disc"])) == "true":
        print("Style disc")
        self.d_loss = self.d_loss + self.d_style_loss
    #====================================================================================================================
    #Identity loss

    self.q_loss = cfg["lambda_latent"] * tf.reduce_mean(tf.square(self.Q_c_given_x - self.z))
    self.d_loss = self.d_loss + self.q_loss
    self.g_loss = self.g_loss + self.q_loss


    self.d_loss_real_sum = scalar_summary("d_loss_real", self.d_loss_real)
    self.d_loss_fake_sum = scalar_summary("d_loss_fake", self.d_loss_fake)
    self.g_loss_sum = scalar_summary("g_loss", self.g_loss)
    self.d_loss_sum = scalar_summary("d_loss", self.d_loss)

    t_vars = tf.trainable_variables()

    self.d_vars = [var for var in t_vars if 'd_' in var.name]
    self.g_vars = [var for var in t_vars if 'g_' in var.name]

    self.saver = tf.train.Saver()

  def train_BlockGAN_multi(self, config):
      self.d_lr_in = tf.placeholder(tf.float32, None, name='d_eta')
      self.g_lr_in = tf.placeholder(tf.float32, None, name='d_eta')

      d_optim = tf.train.AdamOptimizer(cfg['d_eta'], beta1=cfg['beta1'], beta2=cfg['beta2']).minimize(self.d_loss, var_list=self.d_vars)
      g_optim = tf.train.AdamOptimizer(cfg['g_eta'], beta1=cfg['beta1'], beta2=cfg['beta2']).minimize(self.g_loss, var_list=self.g_vars)

      tf.global_variables_initializer().run()

      shutil.copyfile(sys.argv[1], os.path.join(LOGDIR, 'config.json'))

      #Create a new folder to save trained models every 25K step for archive
      ARCHIVE_DIR = os.path.join(LOGDIR, "archive")
      if not os.path.exists(ARCHIVE_DIR):
          os.makedirs(os.path.join(ARCHIVE_DIR))
      self.g_sum = merge_summary([self.d_loss_fake_sum, self.g_loss_sum])
      self.d_sum = merge_summary([self.d_loss_real_sum, self.d_loss_sum])
      self.writer = SummaryWriter(LOGDIR, self.sess.graph)
      skew = self.skew_func(cfg['batch_size'], size=cfg['size'], new_size=cfg['new_size'],
                            focal_length=cfg['focal_length'], sensor_size=cfg['sensor_size'], distance=cfg['cam_dist'])

      # Sample one fixed Z and view parameters to test during training
      sample_z = self.sampling_Z(cfg['z_dim'], str(cfg['sample_z']))
      sample_view = self.gen_view_func(cfg['batch_size'],
                                       cfg['ele_low'], cfg['ele_high'],
                                       cfg['azi_low'], cfg['azi_high'],
                                       cfg['scale_low'], cfg['scale_high'],
                                       cfg['x_low'], cfg['x_high'],
                                       cfg['y_low'], cfg['y_high'],
                                       cfg['z_low'], cfg['z_high'],
                                       with_translation=cfg['with_translation'],
                                       with_scale=to_bool(str(cfg['with_scale'])))

      sample_z2 = self.sampling_Z(cfg['z_dim2'], str(cfg['sample_z']))
      sample_view2 = self.gen_view_func(cfg['batch_size'],
                                       cfg['ele_low2'], cfg['ele_high2'],
                                       cfg['azi_low2'], cfg['azi_high2'],
                                       cfg['scale_low2'], cfg['scale_high2'],
                                       cfg['x_low2'], cfg['x_high2'],
                                       cfg['y_low2'], cfg['y_high2'],
                                       cfg['z_low2'], cfg['z_high2'],
                                       with_translation=cfg['with_translation'],
                                       with_scale=to_bool(str(cfg['with_scale'])))


      sample_files = self.data[0:cfg['batch_size']]
      sample_images = [get_image(sample_file,
                                input_height=self.input_height,
                                input_width=self.input_width,
                                resize_height=self.output_height,
                                resize_width=self.output_width,
                                crop=False) for sample_file in sample_files]

      counter = 1
      start_time = time.time()
      could_load, checkpoint_counter = self.load(self.checkpoint_dir)
      if could_load:
          counter = checkpoint_counter
          print(" [*] Load SUCCESS")
      else:
          print(" [!] Load failed...")

      self.data = glob.glob(os.path.join(IMAGE_PATH, self.input_fname_pattern))
      d_lr = cfg['d_eta']
      g_lr = cfg['g_eta']
      for epoch in range(cfg['max_epochs']):
          if cfg['reduce_eta']:
              d_lr = d_lr if epoch < cfg['epoch_step'] else d_lr * (cfg['max_epochs'] - epoch) / (cfg['max_epochs'] - cfg['epoch_step'])
              g_lr = g_lr if epoch < cfg['epoch_step'] else g_lr * (cfg['max_epochs'] - epoch) / (cfg['max_epochs'] - cfg['epoch_step'])


          random.shuffle(self.data)
          batch_idxs = min(len(self.data), config.train_size) // cfg['batch_size']

          for idx in range(0, batch_idxs):
              batch_files = self.data[idx * cfg['batch_size']:(idx + 1) * cfg['batch_size']]
              batch_images = [get_image(batch_file,
                                input_height=self.input_height,
                                input_width=self.input_width,
                                resize_height=self.output_height,
                                resize_width=self.output_width,
                                crop=False) for batch_file in batch_files]

              batch_z = self.sampling_Z(cfg['z_dim'], str(cfg['sample_z']))
              batch_view = self.gen_view_func(cfg['batch_size'],
                                       cfg['ele_low'], cfg['ele_high'],
                                       cfg['azi_low'], cfg['azi_high'],
                                       cfg['scale_low'], cfg['scale_high'],
                                       cfg['x_low'], cfg['x_high'],
                                       cfg['y_low'], cfg['y_high'],
                                       cfg['z_low'], cfg['z_high'],
                                       with_translation=to_bool(str(cfg['with_translation'])),
                                       with_scale=to_bool(str(cfg['with_scale'])))

              batch_z2 = self.sampling_Z(cfg['z_dim2'], str(cfg['sample_z']))
              batch_view2 = self.gen_view_func(cfg['batch_size'],
                                       cfg['ele_low2'], cfg['ele_high2'],
                                       cfg['azi_low2'], cfg['azi_high2'],
                                       cfg['scale_low2'], cfg['scale_high2'],
                                       cfg['x_low2'], cfg['x_high2'],
                                       cfg['y_low2'], cfg['y_high2'],
                                       cfg['z_low2'], cfg['z_high2'],
                                       with_translation=to_bool(str(cfg['with_translation'])),
                                       with_scale=to_bool(str(cfg['with_scale'])))

              feed = {self.inputs: batch_images,
                      self.z: batch_z,
                      self.view_in: batch_view,
                      self.z2: batch_z2,
                      self.view_in2: batch_view2,
                      self.skew_matrix_in: skew,
                      self.d_lr_in: d_lr,
                      self.g_lr_in: g_lr}


              # Update D network
              for i in range(cfg['D_update']):
                  print("D update: {0}".format(i))
                  _, summary_str = self.sess.run([d_optim, self.d_sum],feed_dict=feed)
                  self.writer.add_summary(summary_str, counter)
              for i in range(cfg['G_update']):
             # Update G network
                  print("G update: {0}".format(i))
                  _, summary_str = self.sess.run([g_optim, self.g_sum], feed_dict=feed)
                  self.writer.add_summary(summary_str, counter)


              errD_fake = self.d_loss_fake.eval(feed)
              errD_real = self.d_loss_real.eval(feed)
              errG = self.g_loss.eval(feed)


              counter += 1
              print("Epoch: [%2d] [%4d/%4d] time: %4.4f, d_loss: %.8f, g_loss: %.8f, q_loss: %.8f" \
                    % (epoch, idx, batch_idxs,
                       time.time() - start_time, errD_fake + errD_real, errG, 0))

              if np.mod(counter, 1000) == 1:
                  self.save(LOGDIR, ARCHIVE_DIR, counter)
                  feed_eval = {self.inputs: sample_images,
                               self.z: sample_z,
                               self.view_in: sample_view,
                               self.z2: sample_z2,
                               self.view_in2: sample_view2,
                               self.skew_matrix_in: skew,
                               self.d_lr_in: d_lr,
                               self.g_lr_in: g_lr}
                  samples, d_loss, g_loss = self.sess.run(
                      [self.G, self.d_loss, self.g_loss],
                      feed_dict=feed_eval)
                  ren_img = inverse_transform(samples)
                  ren_img = np.clip(255 * ren_img, 0, 255).astype(np.uint8)

                  scipy.misc.imsave(
                  os.path.join(OUTPUT_DIR, "{0}_GAN.png".format(counter)),
                  merge(ren_img, [cfg['batch_size'] // 4, 4]))
                  print("[Sample] d_loss: %.8f, g_loss: %.8f" % (d_loss, g_loss))

  def train_BlockGAN_multi_centerCrop_jitter(self, config):
      self.d_lr_in = tf.placeholder(tf.float32, None, name='d_eta')
      self.g_lr_in = tf.placeholder(tf.float32, None, name='d_eta')

      d_optim = tf.train.AdamOptimizer(cfg['d_eta'], beta1=cfg['beta1'], beta2=cfg['beta2']).minimize(self.d_loss, var_list=self.d_vars)
      g_optim = tf.train.AdamOptimizer(cfg['g_eta'], beta1=cfg['beta1'], beta2=cfg['beta2']).minimize(self.g_loss, var_list=self.g_vars)

      tf.global_variables_initializer().run()

      shutil.copyfile(sys.argv[1], os.path.join(LOGDIR, 'config.json'))
      #Create a new folder to save trained models every 25K step for archive
      ARCHIVE_DIR = os.path.join(LOGDIR, "archive")
      if not os.path.exists(ARCHIVE_DIR):
          os.makedirs(os.path.join(ARCHIVE_DIR))
      self.g_sum = merge_summary([self.d_loss_fake_sum, self.g_loss_sum])
      self.d_sum = merge_summary([self.d_loss_real_sum, self.d_loss_sum])
      self.writer = SummaryWriter(LOGDIR, self.sess.graph)
      skew = self.skew_func(cfg['batch_size'], size=cfg['size'], new_size=cfg['new_size'],
                            focal_length=cfg['focal_length'], sensor_size=cfg['sensor_size'], distance=cfg['cam_dist'])

      # Sample one fixed Z and view parameters to test during training
      sample_z = self.sampling_Z(cfg['z_dim'], str(cfg['sample_z']))
      sample_view = self.gen_view_func(cfg['batch_size'],
                                       cfg['ele_low'], cfg['ele_high'],
                                       cfg['azi_low'], cfg['azi_high'],
                                       cfg['scale_low'], cfg['scale_high'],
                                       cfg['x_low'], cfg['x_high'],
                                       cfg['y_low'], cfg['y_high'],
                                       cfg['z_low'], cfg['z_high'],
                                       with_translation=cfg['with_translation'],
                                       with_scale=to_bool(str(cfg['with_scale'])))

      sample_z2 = self.sampling_Z(cfg['z_dim2'], str(cfg['sample_z']))
      sample_view2 = self.gen_view_func(cfg['batch_size'],
                                       cfg['ele_low2'], cfg['ele_high2'],
                                       cfg['azi_low2'], cfg['azi_high2'],
                                       cfg['scale_low2'], cfg['scale_high2'],
                                       cfg['x_low2'], cfg['x_high2'],
                                       cfg['y_low2'], cfg['y_high2'],
                                       cfg['z_low2'], cfg['z_high2'],
                                       with_translation=cfg['with_translation'],
                                       with_scale=to_bool(str(cfg['with_scale'])))

      # Make sure camera elevation is the same for all objects
      sample_view2[:, 1] = sample_view[:, 1]

      sample_files = self.data[0:cfg['batch_size']]


      sample_images = [get_image_jitter(sample_file,
                                input_height=self.input_height,
                                input_width=self.input_width,
                                resize_height=self.output_height,
                                resize_width=self.output_width,
                                ratio=cfg['crop_ratio'],
                                crop=True) for sample_file in sample_files]

      counter = 1
      start_time = time.time()
      could_load, checkpoint_counter = self.load(self.checkpoint_dir)
      if could_load:
          counter = checkpoint_counter
          print(" [*] Load SUCCESS")
      else:
          print(" [!] Load failed...")

      self.data = glob.glob(os.path.join(IMAGE_PATH, self.input_fname_pattern))
      d_lr = cfg['d_eta']
      g_lr = cfg['g_eta']
      for epoch in range(cfg['max_epochs']):
          if cfg['reduce_eta']:
              d_lr = d_lr if epoch < cfg['epoch_step'] else d_lr * (cfg['max_epochs'] - epoch) / (cfg['max_epochs'] - cfg['epoch_step'])
              g_lr = g_lr if epoch < cfg['epoch_step'] else g_lr * (cfg['max_epochs'] - epoch) / (cfg['max_epochs'] - cfg['epoch_step'])


          random.shuffle(self.data)
          batch_idxs = min(len(self.data), config.train_size) // cfg['batch_size']

          for idx in range(0, batch_idxs):
              batch_files = self.data[idx * cfg['batch_size']:(idx + 1) * cfg['batch_size']]
              batch_images = [get_image_jitter(batch_file,
                                input_height=self.input_height,
                                input_width=self.input_width,
                                resize_height=self.output_height,
                                resize_width=self.output_width,
                                ratio = cfg['crop_ratio'],
                                crop=self.crop) for batch_file in batch_files]

              batch_z = self.sampling_Z(cfg['z_dim'], str(cfg['sample_z']))
              batch_view = self.gen_view_func(cfg['batch_size'],
                                       cfg['ele_low'], cfg['ele_high'],
                                       cfg['azi_low'], cfg['azi_high'],
                                       cfg['scale_low'], cfg['scale_high'],
                                       cfg['x_low'], cfg['x_high'],
                                       cfg['y_low'], cfg['y_high'],
                                       cfg['z_low'], cfg['z_high'],
                                       with_translation=to_bool(str(cfg['with_translation'])),
                                       with_scale=to_bool(str(cfg['with_scale'])))

              batch_z2 = self.sampling_Z(cfg['z_dim2'], str(cfg['sample_z']))
              batch_view2 = self.gen_view_func(cfg['batch_size'],
                                       cfg['ele_low2'], cfg['ele_high2'],
                                       cfg['azi_low2'], cfg['azi_high2'],
                                       cfg['scale_low2'], cfg['scale_high2'],
                                       cfg['x_low2'], cfg['x_high2'],
                                       cfg['y_low2'], cfg['y_high2'],
                                       cfg['z_low2'], cfg['z_high2'],
                                       with_translation=to_bool(str(cfg['with_translation'])),
                                       with_scale=to_bool(str(cfg['with_scale'])))

              #Make sure camera elevation is the same for all objects
              batch_view2[:, 1] = batch_view[:, 1]

              feed = {self.inputs: batch_images,
                      self.z: batch_z,
                      self.view_in: batch_view,
                      self.z2: batch_z2,
                      self.view_in2: batch_view2,
                      self.skew_matrix_in: skew,
                      self.d_lr_in: d_lr,
                      self.g_lr_in: g_lr}

              # Update D network
              for i in range(cfg['D_update']):
                  print("D update: {0}".format(i))
                  _, summary_str = self.sess.run([d_optim, self.d_sum],feed_dict=feed)
                  self.writer.add_summary(summary_str, counter)
              # Update G network
              for i in range(cfg['G_update']):
                  print("G update: {0}".format(i))
                  _, summary_str = self.sess.run([g_optim, self.g_sum], feed_dict=feed)
                  self.writer.add_summary(summary_str, counter)


              errD_fake = self.d_loss_fake.eval(feed)
              errD_real = self.d_loss_real.eval(feed)
              errG = self.g_loss.eval(feed)


              counter += 1
              print("Epoch: [%2d] [%4d/%4d] time: %4.4f, d_loss: %.8f, g_loss: %.8f, q_loss: %.8f" \
                    % (epoch, idx, batch_idxs,
                       time.time() - start_time, errD_fake + errD_real, errG, 0))

              if np.mod(counter, 1000) == 1:
                  self.save(LOGDIR, ARCHIVE_DIR, counter)
                  feed_eval = {self.inputs: sample_images,
                               self.z: sample_z,
                               self.view_in: sample_view,
                               self.z2: sample_z2,
                               self.view_in2: sample_view2,
                               self.skew_matrix_in: skew,
                               self.d_lr_in: d_lr,
                               self.g_lr_in: g_lr}
                  samples, d_loss, g_loss = self.sess.run(
                      [self.G, self.d_loss, self.g_loss],
                      feed_dict=feed_eval)
                  ren_img = inverse_transform(samples)
                  ren_img = np.clip(255 * ren_img, 0, 255).astype(np.uint8)

                  scipy.misc.imsave(
                      os.path.join(OUTPUT_DIR, "{0}_GAN.png".format(counter)),
                      merge(ren_img, [cfg['batch_size'] // 4, 4]))
                  print("[Sample] d_loss: %.8f, g_loss: %.8f" % (d_loss, g_loss))

#=======================================================================================================================

  def objects_pool(self, objects, name="objects_pool"):
      vp = tf.expand_dims(objects[0], 0)
      for v in objects[1:]:
          v = tf.expand_dims(v, 0)
          vp = tf.concat([vp, v], 0)
      vp = tf.reduce_max(vp, [0], name=name)
      return vp

  def sampling_Z(self, z_dim, type="uniform"):
      if str.lower(type) == "uniform":
          return np.random.uniform(-1., 1., (cfg['batch_size'], z_dim))
      else:
          return np.random.normal(0, 1, (cfg['batch_size'], z_dim))

  def linear_classifier(self, features, scope = "lin_class", stddev=0.02, reuse=False):
      with tf.variable_scope(scope) as sc:
          w = tf.get_variable('w', [features.get_shape()[-1], 1],
                              initializer=tf.random_normal_initializer(stddev=stddev))
          b = tf.get_variable('biases', 1, initializer=tf.constant_initializer(0.0))
          logits = tf.matmul(features, w) + b
          return   tf.nn.sigmoid(logits), logits

  def z_mapping_function(self, z, output_channel, scope='z_mapping', act="relu", stddev=0.02):
      with tf.variable_scope(scope) as sc:
          w = tf.get_variable('w', [z.get_shape()[-1], output_channel * 2],
                              initializer=tf.random_normal_initializer(stddev=stddev))
          b = tf.get_variable('biases', output_channel * 2, initializer=tf.constant_initializer(0.0))
          if act == "relu":
              out = tf.nn.relu(tf.matmul(z, w) + b)
          else:
              out = lrelu(tf.matmul(z, w) + b)
          return out[:, :output_channel], out[:, output_channel:]

#=======================================================================================================================
##Discriminators
#=======================================================================================================================

  def discriminator_IN(self, image,  cont_dim, reuse=False):
      with tf.variable_scope("discriminator") as scope:
          if reuse:
              scope.reuse_variables()

          h0 = lrelu(conv2d(image, self.df_dim, name='d_h0_conv'))
          h1 = lrelu(instance_norm(conv2d_specNorm(h0, self.df_dim * 2, name='d_h1_conv'),'d_in1'))
          h2 = lrelu(instance_norm(conv2d_specNorm(h1, self.df_dim * 4, name='d_h2_conv'),'d_in2'))
          h3 = lrelu(instance_norm(conv2d_specNorm(h2, self.df_dim * 8, name='d_h3_conv'),'d_in3'))

          #Returning logits to determine whether the images are real or fake
          h4 = linear(slim.flatten(h3), 1, 'd_h4_lin')

          # Recognition network for latent variables has an additional layer
          encoder = lrelu((linear(slim.flatten(h3), 128, 'd_latent')))
          # Compute mean and variance for Gaussian posterior of continuous latents
          cont_vars = linear(encoder, cont_dim, "d_latent_prediction")

          return tf.nn.sigmoid(h4), h4, tf.nn.tanh(cont_vars)

  def discriminator_IN_style_res64(self, image,  cont_dim, reuse=False):
      batch_size = tf.shape(image)[0]
      with tf.variable_scope("discriminator") as scope:
          if reuse:
              scope.reuse_variables()

          h0 = lrelu(conv2d(image, self.df_dim, name='d_h0_conv'))

          h1 = conv2d_specNorm(h0, self.df_dim * 2, name='d_h1_conv')
          h1, h1_mean, h1_var = instance_norm(h1, 'd_in1', True)
          h1_mean = tf.reshape(h1_mean, (batch_size, self.df_dim * 2))
          h1_var = tf.reshape(h1_var, (batch_size, self.df_dim * 2))
          d_h1_style = tf.concat([h1_mean, h1_var], 0)
          d_h1, d_h1_logits = self.linear_classifier(d_h1_style, "d_h1_class")
          h1 = lrelu(h1)

          h2 = conv2d_specNorm(h1, self.df_dim * 4, name='d_h2_conv')
          h2, h2_mean, h2_var = instance_norm(h2, 'd_in2', True)
          h2_mean = tf.reshape(h2_mean, (batch_size, self.df_dim * 4))
          h2_var = tf.reshape(h2_var, (batch_size, self.df_dim * 4))
          d_h2_style = tf.concat([h2_mean, h2_var], 0)
          d_h2, d_h2_logits = self.linear_classifier(d_h2_style, "d_h2_class")
          h2 = lrelu(h2)

          h3 = conv2d_specNorm(h2, self.df_dim * 8, name='d_h3_conv')
          h3, h3_mean, h3_var = instance_norm(h3, 'd_in3', True)
          h3_mean = tf.reshape(h3_mean, (batch_size, self.df_dim * 8))
          h3_var = tf.reshape(h3_var, (batch_size, self.df_dim * 8))
          d_h3_style = tf.concat([h3_mean, h3_var], 0)
          d_h3, d_h3_logits = self.linear_classifier(d_h3_style, "d_h3_class")
          h3 = lrelu(h3)

          # h4 = conv2d_specNorm(h3, self.df_dim * 16, name='d_h4_conv')
          # h4, h4_mean, h4_var = instance_norm(h4, 'd_in4', True)
          # h4_mean = tf.reshape(h4_mean, (batch_size, self.df_dim * 16))
          # h4_var = tf.reshape(h4_var, (batch_size, self.df_dim * 16))
          # d_h4_style = tf.concat([h4_mean, h4_var], 0)
          # d_h4, d_h4_logits = self.linear_classifier(d_h4_style, "d_h4_class")
          # h4 = lrelu(h4)

          #Returning logits to determine whether the images are real or fake
          h5 = linear(slim.flatten(h3), 1, 'd_h5_lin')

          # Recognition network for latent variables has an additional layer
          encoder = lrelu((linear(slim.flatten(h3), 128, 'd_latent')))
          # Compute mean and variance for Gaussian posterior of continuous latents
          cont_vars = linear(encoder, cont_dim, "d_latent_prediction")

          return tf.nn.sigmoid(h5), h5, tf.nn.tanh(cont_vars), d_h1_logits, d_h2_logits, d_h3_logits

  def discriminator_IN_style_res128(self, image,  cont_dim, reuse=False):
      batch_size = tf.shape(image)[0]
      if str(cfg["add_D_noise"]) == "true":
          # Adapted from https://colab.research.google.com/drive/1JkCI_n2U2i6DFU8NKk3P6EkPo3ZTKAaq#scrollTo=eGoqXMJFoWwe
          image = image + tf.random_normal(tf.shape(image), stddev=0.02)

      with tf.variable_scope("discriminator") as scope:
          if reuse:
              scope.reuse_variables()

          h0 = lrelu(conv2d(image, self.df_dim, name='d_h0_conv'))

          h1 = conv2d_specNorm(h0, self.df_dim * 2, name='d_h1_conv')
          h1, h1_mean, h1_var = instance_norm(h1, 'd_in1', True)
          h1_mean = tf.reshape(h1_mean, (batch_size, self.df_dim * 2))
          h1_var = tf.reshape(h1_var, (batch_size, self.df_dim * 2))
          d_h1_style = tf.concat([h1_mean, h1_var], 0)
          d_h1, d_h1_logits = self.linear_classifier(d_h1_style, "d_h1_class")
          h1 = lrelu(h1)

          h2 = conv2d_specNorm(h1, self.df_dim * 4, name='d_h2_conv')
          h2, h2_mean, h2_var = instance_norm(h2, 'd_in2', True)
          h2_mean = tf.reshape(h2_mean, (batch_size, self.df_dim * 4))
          h2_var = tf.reshape(h2_var, (batch_size, self.df_dim * 4))
          d_h2_style = tf.concat([h2_mean, h2_var], 0)
          d_h2, d_h2_logits = self.linear_classifier(d_h2_style, "d_h2_class")
          h2 = lrelu(h2)

          h3 = conv2d_specNorm(h2, self.df_dim * 8, name='d_h3_conv')
          h3, h3_mean, h3_var = instance_norm(h3, 'd_in3', True)
          h3_mean = tf.reshape(h3_mean, (batch_size, self.df_dim * 8))
          h3_var = tf.reshape(h3_var, (batch_size, self.df_dim * 8))
          d_h3_style = tf.concat([h3_mean, h3_var], 0)
          d_h3, d_h3_logits = self.linear_classifier(d_h3_style, "d_h3_class")
          h3 = lrelu(h3)

          h4 = conv2d_specNorm(h3, self.df_dim * 16, name='d_h4_conv')
          h4, h4_mean, h4_var = instance_norm(h4, 'd_in4', True)
          h4_mean = tf.reshape(h4_mean, (batch_size, self.df_dim * 16))
          h4_var = tf.reshape(h4_var, (batch_size, self.df_dim * 16))
          d_h4_style = tf.concat([h4_mean, h4_var], 0)
          d_h4, d_h4_logits = self.linear_classifier(d_h4_style, "d_h4_class")
          h4 = lrelu(h4)

          #Returning logits to determine whether the images are real or fake
          h5 = linear(slim.flatten(h4), 1, 'd_h5_lin')

          # Recognition network for latent variables has an additional layer
          encoder = lrelu((linear(slim.flatten(h4), 128, 'd_latent')))
          # Compute mean and variance for Gaussian posterior of continuous latents
          cont_vars = linear(encoder, cont_dim, "d_latent_prediction")

          return tf.nn.sigmoid(h5), h5, tf.nn.tanh(cont_vars), d_h1_logits, d_h2_logits, d_h3_logits, d_h4_logits

#=======================================================================================================================
##Object generators
#=======================================================================================================================

  def generator_AdaIN_3DFeat_rotate_constantx8(self, z, view_in, skew_matrix, name='generator_feat', reuse=False,
                                               act="relu"):
      # Based on A Style-Based Generator Architecture for Generative Adversarial Networks
      batch_size = tf.shape(z)[0]
      s_h, s_w, s_d = 64, 64, 64  # Depth dimension is the same with the height and width dimension
      s_h2, s_w2, s_d2 = conv_out_size_same(s_h, 2), conv_out_size_same(s_w, 2), conv_out_size_same(s_d, 2)
      s_h4, s_w4, s_d4 = conv_out_size_same(s_h2, 2), conv_out_size_same(s_w2, 2), conv_out_size_same(s_d2, 2)
      s_h8, s_w8, s_d8 = conv_out_size_same(s_h4, 2), conv_out_size_same(s_w4, 2), conv_out_size_same(s_d4, 2)
      s_h16, s_w16, s_d16 = conv_out_size_same(s_h8, 2), conv_out_size_same(s_w8, 2), conv_out_size_same(s_d8, 2)
      if act == "relu":
          act_fn = tf.nn.relu
      else:
          act_fn = lrelu
      with tf.variable_scope(name) as scope:
          if reuse:
              scope.reuse_variables()
          # A learnt constant "template"
          with tf.variable_scope('g_w_constant'):
              w = tf.get_variable('w', [s_d16, s_w16, s_h16, self.gf_dim * 8],
                                  initializer=tf.random_normal_initializer(stddev=0.02))
              w_tile = tf.tile(tf.expand_dims(w, 0),
                               (batch_size, 1, 1, 1, 1))  # Repeat the learnt constant features to make a batch
              s0, b0 = self.z_mapping_function(z, self.gf_dim * 8, 'g_z0')
              h0 = AdaIn(w_tile, s0, b0)
              h0 = act_fn(h0)

          h1 = deconv3d(h0, [batch_size, s_d8, s_w8, s_h8, self.gf_dim * 2], k_h=3, k_d=3, k_w=3,
                        name='g_h1')  # n_filter = 256
          s1, b1 = self.z_mapping_function(z, self.gf_dim * 2, 'g_z1')
          h1 = AdaIn(h1, s1, b1)
          h1 = act_fn(h1)

          h2 = deconv3d(h1, [batch_size, s_d4, s_w4, s_h4, self.gf_dim * 1], k_h=3, k_d=3, k_w=3,
                        name='g_h2')  # n_filter = 256
          s2, b2 = self.z_mapping_function(z, self.gf_dim * 1, 'g_z2')
          h2 = AdaIn(h2, s2, b2)
          h2 = act_fn(h2)

          h2_rotated = tf_3D_transform_skew(h2, view_in, skew_matrix, 16, 16)

          return h2_rotated

  def generator_AdaIN_3DFeat_rotate_constantx4(self, z, view_in, skew_matrix, name='generator_feat', reuse=False,
                                               act="relu"):
      # Based on A Style-Based Generator Architecture for Generative Adversarial Networks
      batch_size = tf.shape(z)[0]
      s_h, s_w, s_d = 64, 64, 64  # Depth dimension is the same with the height and width dimension
      s_h2, s_w2, s_d2 = conv_out_size_same(s_h, 2), conv_out_size_same(s_w, 2), conv_out_size_same(s_d, 2)
      s_h4, s_w4, s_d4 = conv_out_size_same(s_h2, 2), conv_out_size_same(s_w2, 2), conv_out_size_same(s_d2, 2)
      s_h8, s_w8, s_d8 = conv_out_size_same(s_h4, 2), conv_out_size_same(s_w4, 2), conv_out_size_same(s_d4, 2)
      s_h16, s_w16, s_d16 = conv_out_size_same(s_h8, 2), conv_out_size_same(s_w8, 2), conv_out_size_same(s_d8, 2)
      if act == "relu":
          act_fn = tf.nn.relu
      else:
          act_fn = lrelu
      with tf.variable_scope(name) as scope:
          if reuse:
              scope.reuse_variables()
          # A learnt constant "template"
          with tf.variable_scope('g_w_constant'):
              w = tf.get_variable('w', [s_d16, s_w16, s_h16, self.gf_dim * 4],
                                  initializer=tf.random_normal_initializer(stddev=0.02))
              w_tile = tf.tile(tf.expand_dims(w, 0),
                               (batch_size, 1, 1, 1, 1))  # Repeat the learnt constant features to make a batch
              s0, b0 = self.z_mapping_function(z, self.gf_dim * 4, 'g_z0')
              h0 = AdaIn(w_tile, s0, b0)
              h0 = act_fn(h0)

          h1 = deconv3d(h0, [batch_size, s_d8, s_w8, s_h8, self.gf_dim * 2], k_h=3, k_d=3, k_w=3,
                        name='g_h1')  # n_filter = 256
          s1, b1 = self.z_mapping_function(z, self.gf_dim * 2, 'g_z1')
          h1 = AdaIn(h1, s1, b1)
          h1 = act_fn(h1)

          h2 = deconv3d(h1, [batch_size, s_d4, s_w4, s_h4, self.gf_dim * 1], k_h=3, k_d=3, k_w=3,
                        name='g_h2')  # n_filter = 256
          s2, b2 = self.z_mapping_function(z, self.gf_dim * 1, 'g_z2')
          h2 = AdaIn(h2, s2, b2)
          h2 = act_fn(h2)

          h2_rotated = tf_3D_transform_skew(h2, view_in, skew_matrix, 16, 16)

          return h2_rotated

#=======================================================================================================================
##Generators
#=======================================================================================================================

  def generator_AdaIN_skew_MAX_less2DFeat(self, z, z2, view_in, view_in2, skew_matrix, reuse=False):
      batch_size = tf.shape(z)[0]
      s_h, s_w, s_d = 64, 64, 64  # Depth dimension is the same with the height and width dimension
      s_h2, s_w2, s_d2 = conv_out_size_same(s_h, 2), conv_out_size_same(s_w, 2), conv_out_size_same(s_d, 2)
      s_h4, s_w4, s_d4 = conv_out_size_same(s_h2, 2), conv_out_size_same(s_w2, 2), conv_out_size_same(s_d2, 2)

      with tf.variable_scope("generator") as scope:
          if reuse:
              scope.reuse_variables()

          #3D features
          h_FG1 = self.generator_AdaIN_3DFeat_rotate_constantx8(z, view_in, skew_matrix, "g_FG", False)
          h_BG = self.generator_AdaIN_3DFeat_rotate_constantx4(z2, view_in2, skew_matrix, "g_BG", False)

          # COMBINE
          all_objects = []
          all_objects.append(h_FG1)
          all_objects.append(h_BG)
          h2_rotated = self.objects_pool(all_objects)
          h2_rotated = transform_voxel_to_match_image(h2_rotated)


          # Collapsing Z dimension
          h2_2d = tf.reshape(h2_rotated, [batch_size, s_h4, s_w4, 16 * self.gf_dim])
          # 1X1 convolution
          h3 = deconv2d(h2_2d, [batch_size, s_h4, s_w4, self.gf_dim], k_h=1, k_w=1, d_h=1, d_w=1, name='g_h3')
          h3 = tf.nn.relu(h3)

          #2D features
          h4 = deconv2d(h3, [batch_size, s_h2, s_w2, self.gf_dim], k_h=4, k_w=4, name='g_h4')
          h4 = tf.nn.relu(h4)

          h5 = deconv2d(h4, [batch_size, s_h, s_w, self.gf_dim], k_h=4, k_w=4, name='g_h5')
          h5 = tf.nn.relu(h5)

          h6 = deconv2d(h5, [batch_size, s_h, s_w, self.c_dim], k_h=4, k_w=4, d_h=1, d_w=1, name='g_h6')

          output = tf.nn.tanh(h6, name="output")
          return output

  def generator_AdaIN_skew_MAX_lrelu(self, z, z2, view_in, view_in2, skew_matrix, reuse=False):
      batch_size = tf.shape(z)[0]
      s_h, s_w, s_d = 64, 64, 64  # Depth dimension is the same with the height and width dimension
      s_h2, s_w2, s_d2 = conv_out_size_same(s_h, 2), conv_out_size_same(s_w, 2), conv_out_size_same(s_d, 2)
      s_h4, s_w4, s_d4 = conv_out_size_same(s_h2, 2), conv_out_size_same(s_w2, 2), conv_out_size_same(s_d2, 2)

      with tf.variable_scope("generator") as scope:
          if reuse:
              scope.reuse_variables()

          #3D features
          h_FG1 = self.generator_AdaIN_3DFeat_rotate_constantx8(z, view_in, skew_matrix, "g_FG", False, 'lrelu')
          h_BG = self.generator_AdaIN_3DFeat_rotate_constantx4(z2, view_in2, skew_matrix, "g_BG", False, 'lrelu')

          # COMBINE
          all_objects = []
          all_objects.append(h_FG1)
          all_objects.append(h_BG)
          h2_rotated = self.objects_pool(all_objects)
          h2_rotated = transform_voxel_to_match_image(h2_rotated)
          # Collapsing Z dimension
          h2_2d = tf.reshape(h2_rotated, [batch_size, s_h4, s_w4, 16 * self.gf_dim])
          # 1X1 convolution
          h3 = deconv2d(h2_2d, [batch_size, s_h4, s_w4, self.gf_dim * 4], k_h=1, k_w=1, d_h=1, d_w=1, name='g_h3')
          h3 = lrelu(h3)

          #2D features
          h4 = deconv2d(h3, [batch_size, s_h2, s_w2, self.gf_dim * 2], k_h=4, k_w=4, name='g_h4')
          h4 = lrelu(h4)

          h5 = deconv2d(h4, [batch_size, s_h, s_w, self.gf_dim], k_h=4, k_w=4, name='g_h5')
          h5 = lrelu(h5)

          h6 = deconv2d(h5, [batch_size, s_h, s_w, self.c_dim], k_h=4, k_w=4, d_h=1, d_w=1, name='g_h6')

          output = tf.nn.tanh(h6, name="output")
          return output

  def generator_AdaIN_skew_MAX_HoloGAN_fixed_weakerBG_lrelu_res128(self, z, z2, view_in, view_in2, skew_matrix, reuse=False):
      batch_size = tf.shape(z)[0]
      s_h, s_w, s_d = 64, 64, 64  # Depth dimension is the same with the height and width dimension
      s_h2, s_w2, s_d2 = conv_out_size_same(s_h, 2), conv_out_size_same(s_w, 2), conv_out_size_same(s_d, 2)
      s_h4, s_w4, s_d4 = conv_out_size_same(s_h2, 2), conv_out_size_same(s_w2, 2), conv_out_size_same(s_d2, 2)
      s_h8, s_w8, s_d8 = conv_out_size_same(s_h4, 2), conv_out_size_same(s_w4, 2), conv_out_size_same(s_d4, 2)
      s_h16, s_w16, s_d16 = conv_out_size_same(s_h8, 2), conv_out_size_same(s_w8, 2), conv_out_size_same(s_d8, 2)
      with tf.variable_scope("generator") as scope:
          if reuse:
              scope.reuse_variables()

          h_FG1 =  self.generator_AdaIN_3DFeat_rotate_constantx8(z, view_in, skew_matrix, "g_FG", False, 'lrelu')
          h_BG = self.generator_AdaIN_3DFeat_rotate_constantx4(z2, view_in2, skew_matrix, "g_BG", False, 'lrelu')

          #COMBINE
          all_objects =[]
          all_objects.append(h_FG1)
          all_objects.append(h_BG)
          h2_rotated = self.objects_pool(all_objects)
          h2_rotated = transform_voxel_to_match_image(h2_rotated)
          # Collapsing Z dimension
          h2_2d = tf.reshape(h2_rotated, [batch_size, s_h4, s_w4, 16 * self.gf_dim])
          # 1X1 convolution
          h3 = deconv2d(h2_2d, [batch_size, s_h4, s_w4, self.gf_dim * 4], k_h=1, k_w=1, d_h=1, d_w=1, name='g_h3')
          h3 = lrelu(h3)

          h4 = deconv2d(h3, [batch_size, s_h2, s_w2, self.gf_dim * 2], k_h=4, k_w=4, name='g_h4')
          h4 = lrelu(h4)

          h4_1 = deconv2d(h4, [batch_size, s_h, s_w, self.gf_dim], k_h=4, k_w=4, name='g_h4_1')
          h4_1 = lrelu(h4_1)

          h5 = deconv2d(h4_1, [batch_size, s_h * 2, s_w * 2, self.gf_dim // 2], k_h=4, k_w=4, name='g_h5')
          h5 = lrelu(h5)

          h6 = deconv2d(h5, [batch_size, s_h * 2, s_w * 2, self.c_dim], k_h=4, k_w=4, d_h=1, d_w=1, name='g_h6')

          output = tf.nn.tanh(h6, name="output")
          return output
#==================================================================================================================

  @property
  def model_dir(self):
    return "{}_{}_{}".format(
        self.dataset_name,
        self.output_height, self.output_width)

  def save(self, checkpoint_dir, archive_dir, step):
    model_name = "HoloGAN.model"
    checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

    if not os.path.exists(checkpoint_dir):
      os.makedirs(checkpoint_dir)

    self.saver.save(self.sess,
            os.path.join(checkpoint_dir, model_name),
            global_step=step)
    if np.mod(step, 25000) == 1:
        all_files = glob.glob(os.path.join(checkpoint_dir, "{0}-{1}*".format(model_name, step)))
        for file in all_files:
            shutil.copy(file, os.path.join(archive_dir, os.path.basename(file)))

  def load(self, checkpoint_dir):
    import re
    print(" [*] Reading checkpoints...")
    checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
      ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
      self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
      counter = int(next(re.finditer("(\d+)(?!.*\d)",ckpt_name)).group(0))
      print(" [*] Success to read {}".format(ckpt_name))
      return True, counter
    else:
      print(" [*] Failed to find a checkpoint")
      return False, 0


