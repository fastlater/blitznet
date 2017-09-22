from glob import glob
import logging
import logging.config
import os

import tensorflow as tf
import numpy as np
from PIL import ImageFont

from config import get_logging_config, args, evaluation_logfile
from config import config as net_config
from paths import CKPT_ROOT

import matplotlib
matplotlib.use('Agg')

from vgg import VGG
from resnet import ResNet
from voc_loader import VOCLoader
from new_loader import NEWLoader
from evaluation import Evaluation
from detector import Detector

slim = tf.contrib.slim

logging.config.dictConfig(get_logging_config(args.run_name))
log = logging.getLogger()



def main(argv=None):
  
    net = ResNet(config=net_config, depth=50, training=False)
    print('[CONSOLE]: Resnet configured.')
    #loader = VOCLoader('12', 'val', segmentation=args.segment)# year, file, augSeg
    #print('[CONSOLE]: VOC12 loaded.')
    loader = NEWLoader('val',False, False)
    
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                          log_device_placement=False)) as sess:
        print('[CONSOLE]: Session started.')
        detector = Detector(sess, net, loader, net_config, no_gt=args.no_seg_gt)
        detector.restore_from_ckpt(args.ckpt)
        name = 'raccoon-81'
        img = loader.load_image(name)
        print('[CONSOLE]: Image ',name, ' loaded.')
        h,w, d = img.shape
        seg_gt = np.zeros([h, w], dtype=np.uint8)
        gt_bboxes = []
        gt_cats = []
        output = detector.feed_forward(img, seg_gt, w, h, name,
                                            gt_bboxes,gt_cats,
                                            True)#If true, draw
        det_bboxes, det_probs, det_cats = output[:3]
        print('det_bboxes: ', det_bboxes)# [xmin, ymin, width, heigth]
        print('det_probs: ', det_probs)# probabiliy [0 ~ 0.99] of each of the det_bboxes
        print('det_cats: ', det_cats) # class [0 ~ 20] of each of the det_bboxes
        print('Results saved at Results/',args.run_name)


if __name__ == '__main__':
    tf.app.run()
