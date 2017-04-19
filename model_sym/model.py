# -*- coding=utf-8 -*-
"""contains instructions for simulation model construction"""

import csv
from datetime import datetime
import numpy as np
import tensorflow as tf
import ann_tf_classes_stripped as ann

SKIPSTEP = 10  # for reducing set size

data = np.array([])
with open('../data_ANN_full.csv') as datafile:
    reader = csv.reader(datafile)
    for row in reader:
        data = np.append(data, row)
data = data.reshape(23, 20045)

data_names = np.array([])
with open('../data_ANN_full_names.csv') as datafile:
    reader = csv.reader(datafile)
    for row in reader:
        data_names = np.append(data_names, row)

data_dict = dict(zip(data_names, data))
indx = np.arange(data_dict['B'].size)
np.random.shuffle(indx)

T0_l = np.array([])
P0_l = np.array([])
Gt_l = np.array([])
T0_t = np.array([])
P0_t = np.array([])
Gt_t = np.array([])
T0_v = np.array([])
P0_v = np.array([])
Gt_v = np.array([])
# learning set
for ind in indx[:int(np.floor(indx.size*0.7)):SKIPSTEP]:
    T0_l = np.append(T0_l, data_dict['T0cp'][ind])
    P0_l = np.append(P0_l, data_dict['B'][ind])
    Gt_l = np.append(Gt_l, data_dict['GT1'][ind])
# testing set
for ind in indx[
        int(np.floor(indx.size*0.7)):int(np.floor(indx.size*0.9)):SKIPSTEP]:
    T0_t = np.append(T0_t, data_dict['T0cp'][ind])
    P0_t = np.append(P0_t, data_dict['B'][ind])
    Gt_t = np.append(Gt_t, data_dict['GT1'][ind])
# validation set
for ind in indx[int(np.floor(indx.size*0.9))::SKIPSTEP]:
    T0_v = np.append(T0_v, data_dict['T0cp'][ind])
    P0_v = np.append(P0_v, data_dict['B'][ind])
    Gt_v = np.append(Gt_v, data_dict['GT1'][ind])

INPUTS = [
        tf.placeholder(tf.float32, name=item) for item in ['T0', 'P0', 'Gt']]
TT = [tf.placeholder(tf.float32, name=item) for item in [
    'MGPower', 'OutPower', 'T4', 'EngSpeed']]

mln_layout = [5 for _ in range(5)]
mln_layout[-1] = len(ann.TT)
mln_sym = ann.NMLNetwork(
        inputs=INPUTS, tt=TT, layout_list = mln_layout, name='sym')
# mln_out = mln_sym.get_out()

# with tf.name_scope('errors'):
    # error_sq = tf.squared_difference(ann.TT, mln_out)
    # error_exp = 1 - tf.exp(-tf.squared_difference(ann.TT, mln_out))
    # error_abs = (ann.TT - mln_out)/(1 + tf.abs(ann.TT - mln_out))
    # error_root = [
    #         tf.sqrt(
    #             tf.squared_difference(TT, out) + 1) - 1 for out in mln_out]
    # mln_err = tf.add_n(error_root)
    # tf.summary.scalar('error_sq', error_sq)
    # tf.summary.scalar('error_exp', error_exp)
    # tf.summary.scalar('error_abs', error_abs)
    # tf.summary.scalar('error_root', error_root)
# tf.summary.scalar('output', mln_out)
# mergsumm = tf.summary.merge_all()

optim = tf.train.MomentumOptimizer(0.01, 0.9)
# optim = tf.train.GradientDescentOptimizer(0.1)
train_step = optim.minimize(mln_sym.errsum)
sess = tf.Session()
sess.run(tf.global_variables_initializer())
curtime = datetime.strftime(datetime.now(), '%H%M%S')
WRITER = tf.summary.FileWriter('tbrd_model_sym/' + curtime, sess.graph)
