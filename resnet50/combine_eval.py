from __future__ import print_function
import argparse
import os
import sys
import time
import cv2

from PIL import Image
import tensorflow as tf
import numpy as np
from scipy import misc
from tqdm import trange
from model_fine import PSPNet101
from model_coarse import PSPNet101_2
from tools import *
import util
import helpers
from image_reader import ImageReader



cityscapes_param = {'crop_size': [720, 720],
                    'num_classes': 19,
                    'ignore_label': 255,
                    'num_steps': 500,
                    'model': PSPNet101,
                    'model2': PSPNet101_2,
                    'data_dir': './cityscapes/',  #### Change this line
                    'val_list': './list/cityscapes_val_list.txt'}


def get_arguments():
    parser = argparse.ArgumentParser(description="Reproduced PSPNet")

    parser.add_argument("--flipped-eval", action="store_true",
                        help="whether to evaluate with flipped img.")
    parser.add_argument("--dataset", type=str, default='cityscapes',
                        choices=['ade20k', 'cityscapes'])

    return parser.parse_args()


def save(saver, sess, logdir, step):
    model_name = 'model.ckpt'
    checkpoint_path = os.path.join(logdir, model_name)

    if not os.path.exists(logdir):
        os.makedirs(logdir)
    saver.save(sess, checkpoint_path, global_step=step)
    print('The checkpoint has been created.')

def load(saver, sess, ckpt_path):
    saver.restore(sess, ckpt_path)
    print("Restored model parameters from {}".format(ckpt_path))

def load_img(img_path):


    filename = img_path.split('/')[-1]
    img = misc.imread(img_path, mode='RGB')
    # print('input image shape: ', img.shape)

    return img, filename


def main():
    args = get_arguments()

    # load parameters
    param = cityscapes_param

    crop_size = param['crop_size']
    num_classes = param['num_classes']
    ignore_label = param['ignore_label']
    num_steps = param['num_steps']
    PSPNet = param['model']
    PSPNet_2 = param['model2']
    data_dir = param['data_dir']


    image_h = 1024
    image_w = 2048
    overlap_size = 256

    # Set up tf session and initialize variables.
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    g1 = tf.Graph()
    g2 = tf.Graph()
    g3 = tf.Graph()

    sess = tf.Session(config=config, graph=g1)
    sess2 = tf.Session(config=config, graph=g2)
    sess3 = tf.Session(config=config, graph=g3)

    # Create network.1
    with g1.as_default():
        sess.run(tf.global_variables_initializer())

        x = tf.placeholder(dtype=tf.float32, shape=[None, None, 3])
        img = preprocess(x, image_h, image_w)
        img = tf.image.crop_to_bounding_box(img, 0, 0, overlap_size, overlap_size)
        net = PSPNet({'data': img}, is_training=False, num_classes=num_classes)
        raw_output = net.layers['conv6']
        # Predictions.
        raw_output_up = tf.image.resize_bilinear(raw_output, size=[overlap_size, overlap_size], align_corners=True)
        raw_output_up = tf.image.pad_to_bounding_box(raw_output_up, 0, 0, image_h, image_w)
        restore_var = tf.global_variables()

        ckpt = tf.train.get_checkpoint_state("./logdir_fine/")
        if ckpt and ckpt.model_checkpoint_path:
            loader = tf.train.Saver(var_list=restore_var)
            load_step = int(os.path.basename(ckpt.model_checkpoint_path).split('-')[1])
            load(loader, sess, ckpt.model_checkpoint_path)
        else:
            print('No checkpoint file found.')


    # Create network.2
    with g2.as_default():
        sess2.run(tf.global_variables_initializer())

        x2 = tf.placeholder(dtype=tf.float32, shape=[None, None, 3])
        img2 = preprocess(x2, image_h, image_w)
        net2 = PSPNet_2({'data': img2}, is_training=False, num_classes=num_classes)
        raw_output2 = net2.layers['conv6_v2']
        # Predictions.
        raw_output_up2 = tf.image.resize_bilinear(raw_output2, size=[image_h, image_w], align_corners=True)
        raw_output_up2 = tf.image.pad_to_bounding_box(raw_output_up2, 0, 0, image_h, image_w)

        restore_var2 = tf.global_variables()

        ckpt2 = tf.train.get_checkpoint_state('./logdir_coarse/')
        if ckpt2 and ckpt2.model_checkpoint_path:
            loader2 = tf.train.Saver(var_list=restore_var2)
            load(loader2, sess2, ckpt2.model_checkpoint_path)
        else:
            print('No checkpoint file found.')

    # combine
    with g3.as_default():
        model1 = tf.placeholder(dtype=tf.float32, shape=[None, None, None, 19])
        model2 = tf.placeholder(dtype=tf.float32, shape=[None, None, None, 19])

        Weights1 = tf.Variable(initial_value=[0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5])
        Weights2 = tf.Variable(initial_value=[0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5])

        combine_output = tf.add(tf.multiply(model1,Weights1),tf.multiply(model2,Weights2))

        sess3.run(tf.global_variables_initializer())
        ckpt3 = tf.train.get_checkpoint_state('./combine_variables/')
        if ckpt3 and ckpt3.model_checkpoint_path:
            loader3 = tf.train.Saver()
            load(loader3, sess3, ckpt3.model_checkpoint_path)
        else:
            print('No checkpoint file found.')

    anno_filename = tf.placeholder(dtype=tf.string)
    # Read & Decode image
    anno = tf.image.decode_image(tf.read_file(anno_filename), channels=1)
    anno.set_shape([None, None, 1])
    pred_placeholder = tf.placeholder(dtype=tf.int64)

    pred_expand = tf.expand_dims(pred_placeholder, dim=3)
    pred_flatten = tf.reshape(pred_expand, [-1, ])
    raw_gt = tf.reshape(anno, [-1, ])
    indices = tf.squeeze(tf.where(tf.not_equal(raw_gt, ignore_label)), 1)
    gt = tf.cast(tf.gather(raw_gt, indices), tf.int32)
    pred = tf.gather(pred_flatten, indices)

    mIoU, update_op = tf.contrib.metrics.streaming_mean_iou(pred, gt, num_classes=num_classes)

    eval_sess = tf.Session(config=config)
    eval_sess.run(tf.local_variables_initializer())

    class_scores_list = []
    class_names_list, label_values = helpers.get_label_info("class_dict.csv")
    file = open(param['val_list'], 'r')
    file_RGB = open("./list/cityscapes_val_list_RGB.txt", 'r')
    for step in trange(num_steps, desc='evaluation', leave=True):
        f1, f2 = file.readline().split('\n')[0].split(' ')
        f1 = os.path.join(data_dir, f1)
        f2 = os.path.join(data_dir, f2)

        img_out, _ = load_img(f1)

        # model 1

        preds1 = np.zeros((1, 1024, 2048, 19))
        for over_h in range(0, image_h, 256):
            for over_w in range(0, image_w, 256):
                if over_h <= image_h - overlap_size and over_w <= image_w - overlap_size:

                    pred_imgs=img_out[over_h:over_h+overlap_size, over_w:over_w+overlap_size]
                    tmp = sess.run(raw_output_up, feed_dict={x: pred_imgs})
                    tmp2=np.zeros((1,1024,2048,19))
                    for i in range(overlap_size):
                        for j in range(overlap_size):
                            tmp2[0][i+over_h][j+over_w]=tmp[0][i][j]
                    preds1 += tmp2

        preds2 = sess2.run(raw_output_up2, feed_dict={x2: img_out})

        combine_out = sess3.run([combine_output], feed_dict={model1: preds1, model2: preds2})

        img_out = np.argmax(combine_out[0], axis=3)
        preds = helpers.colour_code_segmentation(img_out, label_values)
        # misc.imsave("./1106test/" + str(step) + ".png", preds[0])

        # Average per class test accuracies
        RGB_label = file_RGB.readline().split('\n')[0].split(' ')[1]
        RGB_label = os.path.join(data_dir, RGB_label)
        img_label, _ = load_img(RGB_label)
        img_label = helpers.reverse_one_hot(helpers.one_hot_it(img_label, label_values))
        class_accuracies = util.evaluate_segmentation(pred=img_out, label=img_label, num_classes=num_classes)
        class_scores_list.append(class_accuracies)

        _ = eval_sess.run(update_op, feed_dict={pred_placeholder: img_out, anno_filename: f2})
        print('mIoU: {:04f}'.format(eval_sess.run(mIoU)))

    class_avg_scores = np.mean(class_scores_list, axis=0)
    print("Average per class test accuracies = \n")
    for index, item in enumerate(class_avg_scores):
        print("%s = %f" % (class_names_list[index], item))
    print('mIoU: {:04f}'.format(eval_sess.run(mIoU)))

if __name__ == '__main__':
    main()
