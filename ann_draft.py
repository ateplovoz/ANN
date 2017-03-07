# -* coding=utf-8 -*-
# Рабочий скрипт нейросети

import csv
import numpy as np
import tensorflow as tf

data = np.array([])

with open('data_m.csv') as datafile:
    reader = csv.reader(datafile)
    for row in reader:
        data = np.append(data, row)
data = data.reshape(3,32).astype(float)  # Три 1-тензора длиной 32
data[0] = data[0]*0.001  # Перевод Вт в кВт

# Формируется архив индексов, который потом перемешивается.
# Обучающей, тестовой и валидационной выборке присваиваются одинаковые
# значения. Прирчём соотношения выборок 70:20:10
indx = np.arange(data[0].size)
np.random.shuffle(indx)
n_learn = np.array([])
gt_learn = np.array([])
gv_learn = np.array([])
n_test = np.array([])
gt_test = np.array([])
gv_test = np.array([])
n_val = np.array([])
gt_val = np.array([])
gv_val = np.array([])

for ind in indx[:int(np.floor(indx.size*0.7))]:
    n_learn = np.append(n_learn, data[0][ind])
    gv_learn = np.append(gv_learn, data[1][ind])
    gt_learn = np.append(gt_learn, data[2][ind])
for ind in indx[int(np.floor(indx.size*0.7)):int(np.floor(indx.size*0.9))]:
    n_test = np.append(n_test, data[0][ind])
    gv_test = np.append(gv_test, data[1][ind])
    gt_test = np.append(gt_test, data[2][ind])
for ind in indx[int(np.floor(indx.size*0.9)):]:
    n_val = np.append(n_val, data[0][ind])
    gv_val = np.append(gv_val, data[1][ind])
    gt_val = np.append(gt_val, data[2][ind])

pre_ii = [gv_learn, gt_learn]
pre_ii_names = ['gv', 'gt']
with tf.name_scope('ii') as scope:
    ii = [tf.placeholder(tf.float32, name=src_n) for src_n in pre_ii_names] + [
            tf.constant(1.0, name='constant')]
pre_tt = [n_learn]
pre_tt_names = ['n']
with tf.name_scope('tt'):
    tt = [tf.placeholder(tf.float32, name=tg_n) for tg_n in pre_tt_names]
layer_size = 5
ww = [0]*layer_size
iws = [0]*layer_size
oo = [0]*layer_size
err = [0]*layer_size
for lneur in range(layer_size):
    with tf.name_scope('Neuron_{}'.format(lneur)) as scope:
        ww[lneur] = [
            tf.Variable(
                np.random.rand(),
                name='neuron_{0}_iw_{1}'.format(lneur, src_n))
                    for src_n in pre_ii_names
            ] + [tf.Variable(np.random.rand(), name='neuron_{}_cw'.format(lneur))]
        iws[lneur] = tf.cumsum(
                tf.multiply(
                ww[lneur], ii,
                name='neuron_{}_iws'.format(lneur)), reverse=True)[0]
        oo[lneur] = tf.nn.relu(iws[lneur], name='neuron_{}_oo'.format(lneur))
    with tf.name_scope('error_{}'.format(lneur)):
        err[lneur] = tf.square(oo[lneur] - tt)

Loo = tf.add_n(oo, name='Loo')
Lerr = tf.add_n(err, name='Lerr')




def form_feeder(feed_who, feed_what, pick):
    """Forms dict_feed by selecting row from arrays

    `feed_who` and `feed_what` should be same length to form pairs
    (sink: source) from dict(zip(...)).

    Args:
        feed_who: `list of Tensor placeholder` to feed data
        feed_what: `iterable`. Data which we feed to `Tensor`
        pick: `int` row which we select from data

    Returns:
        feeder: `dict` appropriate to use as dict_feed to feed placeholders"""

    feeder = {}
    for sink, source in zip(feed_who, feed_what):
        feeder[sink] = source[pick]
    return feeder

# TODO: write learning cycle
