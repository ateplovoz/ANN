# -*- coding=utf-8 -*-
"""contains instructions for simulation model construction"""

import csv
from datetime import datetime
from importlib import reload
import numpy as np
import tensorflow as tf
import ann_tf_classes_stripped as ann

SKIPSTEP = 1  # > 1 for reducing set size

data = np.array([])
with open('data_ANN_uniform.csv') as datafile:
    reader = csv.reader(datafile)
    for row in reader:
        data = np.append(data, row)
data = data.astype(float)
data = data.reshape(700, 7)

indx = np.arange(data[:, 0].size)
np.random.shuffle(indx)

T0_l, P0_l, Gt_l, MGP_l, OP_l, T4_l, ES_l = (np.array([]),)*7
T0_t, P0_t, Gt_t, MGP_t, OP_t, T4_t, ES_t = (np.array([]),)*7
T0_v, P0_v, Gt_v, MGP_v, OP_v, T4_v, ES_v = (np.array([]),)*7

# learning set
for ind in indx[:int(np.floor(indx.size*0.7)):SKIPSTEP]:
    T0_l = np.append(T0_l, data[:, 0][ind])
    P0_l = np.append(P0_l, data[:, 1][ind])
    Gt_l = np.append(Gt_l, data[:, 2][ind])
    MGP_l = np.append(MGP_l, data[:, 3][ind])
    OP_l = np.append(OP_l, data[:, 4][ind])
    T4_l = np.append(T4_l, data[:, 5][ind])
    ES_l = np.append(ES_l, data[:, 6][ind])
MGP_l = MGP_l*0.001 #  [Вт] -> [кВт]
OP_l = OP_l*0.001 #  [Вт] -> [кВт]
ES_l = ES_l*0.001 #  [об/мин] -> [1000*об/мин]
# testing set
for ind in indx[
    int(np.floor(indx.size*0.7)):int(np.floor(indx.size*0.9)):SKIPSTEP]:
    T0_t = np.append(T0_t, data[:, 0][ind])
    P0_t = np.append(P0_t, data[:, 1][ind])
    Gt_t = np.append(Gt_t, data[:, 2][ind])
    MGP_t = np.append(MGP_t, data[:, 3][ind])
    OP_t = np.append(OP_t, data[:, 4][ind])
    T4_t = np.append(T4_t, data[:, 5][ind])
    ES_t = np.append(ES_t, data[:, 6][ind])
MGP_t = MGP_t*0.001 #  [Вт] -> [кВт]
OP_t = OP_t*0.001 #  [Вт] -> [кВт]
ES_t = ES_l*0.001 #  [об/мин] -> [1000*об/мин]
# validation set
for ind in indx[int(np.floor(indx.size*0.9))::SKIPSTEP]:
    T0_v = np.append(T0_v, data[:, 0][ind])
    P0_v = np.append(P0_v, data[:, 1][ind])
    Gt_v = np.append(Gt_v, data[:, 2][ind])
    MGP_v = np.append(MGP_v, data[:, 3][ind])
    OP_v = np.append(OP_v, data[:, 4][ind])
    T4_v = np.append(T4_v, data[:, 5][ind])
    ES_v = np.append(ES_v, data[:, 6][ind])
MGP_v = MGP_t*0.001 #  [Вт] -> [кВт]
OP_v = OP_t*0.001 #  [Вт] -> [кВт]
ES_v = ES_l*0.001 #  [об/мин] -> [1000*об/мин]

learnset = (
    ('T0', T0_l), ('P0', P0_l), ('Gt', Gt_l),
    ('MGP', MGP_l), ('OP', OP_l), ('T4', T4_l), ('ES', ES_l))
testset = (
    ('T0', T0_t), ('P0', P0_t), ('Gt', Gt_t),
    ('MGP', MGP_t), ('OP', OP_t), ('T4', T4_t), ('ES', ES_t))
valset = (
    ('T0', T0_v), ('P0', P0_v), ('Gt', Gt_v),
    ('MGP', MGP_v), ('OP', OP_v), ('T4', T4_v), ('ES', ES_v))
sortedset = (
    ('T0', data[0]), ('P0', data[1]),
    ('Gt', data[2]), ('MGP', data[3]),
    ('OP', data[4]), ('T4', data[5]),
    ('ES', data[6]))

DSOCKET = ann.Datasocket(*learnset)

mln_layout = [10 for _ in range(6)]
mln_sym = ann.NMLNetwork(
        inputs=DSOCKET.get_sock('T0', 'P0', 'Gt'),
        tt=DSOCKET.get_sock('MGP', 'OP', 'T4', 'ES'),
        layout_list = mln_layout, name='sym')

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
