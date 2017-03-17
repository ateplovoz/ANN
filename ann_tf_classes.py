# -* coding=utf-8 -*-
# Рабочий скрипт нейросети
"""Contains tensorflow ann classes and functions for data learning"""

import csv
import io
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


class NNeuron():
    """Neuron class.

    Neuron is a basic element of neural network. Can have multiple inputs, but
    only one output."""

    def __init__(self, inputs, name=None):
        """Neuron class --- basic neuron

        Creates a neuron with relu activation function

        Args:
            inputs: `list of Tensor`. Tensors that we connect to the neuron
            type: `str='relu` or any other `str`. type of activation function
            name: `str`. Name of tensor node.

        Returns:
            object of NNeuron class

        Raises:
            `TypeError` if inputs are not iterable"""
        try:
            iter(inputs)
        except TypeError:
            raise TypeError('inputs must be iterable')
        self.default_name = 'neuron'
        if not name:
            self.name = self.default_name
        else:
            self.name = name
        self.iws = None
        self.oo = None
        with tf.name_scope(self.name):
            with tf.name_scope('ww'):
                self.ww = [
                    tf.Variable(
                        np.random.rand(),
                        name='ww_{}'.format(item))
                    for item in np.arange(len(inputs))]
            with tf.name_scope('iws'):
                self.iw = [
                    tf.multiply(in_it, ww_it, name='iw')
                    for in_it, ww_it in zip(inputs, self.ww)]
                self.iws = tf.add_n(self.iw, name='sum_of_iw')
            with tf.name_scope('oo'):
                self.oo = tf.nn.relu(self.iws)

    def tune_weight(self, val, randval):
        """Tunes weights of neuron

        This function sets each weight to given value and after that adds
        random value (pos or neg) within given range.

        Args:
            value: `float` to which value set neurons
            rand: `float` radius in which final value can be randomized"""
        with tf.name_scope(self.name):
            with tf.name_scope('ww'):
                self.ww = [
                    tf.Variable(
                        val + np.random.rand()*randval,
                        name='ww_{}'.format(item))
                    for item in np.arange(len(inputs))]



class NLayer():
    """Layer of neurons (`NNeuron`s)

    Contains array of `NNeuron`s and connects inputs to given tensors"""

    def __init__(self, inputs, num_neurons, name=None):
        """Neural Layer class --- creates layer of neurons

        Forms a layer of neurons with given number of neurons, each one having
        given number of inputs plus one (for a constant input)
        Args:
            inputs: `list of Tensor` inputs connected to the layer
            num_neurons: `int`. Number of neurons inside layer
            name: `str`. Name scope of layer

        Returns:
            object of NLayer class

        Raises:
            TypeError if inputs are not iterable
        """
        try:
            iter(inputs)
        except TypeError:
            raise TypeError('inputs must be iterable')
        self.default_name = 'Layer'
        if not name:
            self.name = self.default_name
        else:
            self.name = name

        with tf.name_scope(self.name):
            # Creates tf.constant inside of layer and injects it into inputs
            # as constant input (duh)
            self.inputs = inputs + [tf.constant(1.0, name='const_input')]
            self.VN = [
                NNeuron(
                    self.inputs,
                    name='{0}_neur{1}'.format(self.name, neur_c))
                for neur_c in range(num_neurons)]

    def get_out(self):
        """returns output of layer"""
        return [neur.oo for neur in self.VN]

    def tune_weight(self, val, randval):
        """Tunes weight for each neuron in layer

        Calls `tune_weight` method from NNeuron objects, passes given
        arguments.

        Args:
            value: `float` to which value set neurons
            rand: `float` radius in which final value can be randomized"""
        for neur in self.VN:
            neur.tune_weight(val, randval)


class NMLNetwork():
    """Multi-layer neural network.

    Multi-layer network contains several layers, containing multiple neurons
    each. Only first layer is connected to inputs. Outputs of last layer can
    be summarized to network output."""
    def __init__(self, inputs, layout_list, name=None):
        """Neural multi-layer network --- creates multi-layered network

        Creates network structure according to `layout_list`

        Args:
            inputs: `Tensors` connected to the first layer
            layout_list: `iterable`. Preferably list of `int` of neurons in
            each layer accordingly. Total layer count equal to len(layout_list)
            name: `str`. name scope of MLN

        Returns:
            object of NMLNetwork class

        Raises:
            TypeError if inputs or layout_list are not iterable
        """
        try:
            iter(inputs)
            iter(layout_list)
        except TypeError:
            raise TypeError('inputs/layout_list must be iterable')
        self.default_name = 'multi-layer'
        if not name:
            self.name = self.default_name
        else:
            self.name = name
        with tf.name_scope(self.name):
            self.LL = [0]*len(layout_list)
            lcount = 1
            self.LL[0] = NLayer(inputs, layout_list[0], name='layer0')
            # probably weird behaviour if `layout_list`
            # is short (less than 2 items)
            for num_neurons in layout_list[1:]:
                self.LL[lcount] = NLayer(
                    self.LL[lcount - 1].get_out(),
                    num_neurons, name='layer{}'.format(lcount))
                lcount += 1

    def get_out(self):
        """returns output of the last layer"""
        return self.LL[-1].get_out()

    def tune_weight(self, val, randval):
        """Sets new weight value for each neuron within network.

        Calls tune_weight method from NLayer objects. Passes given arguments.
        Args:
            value: `float` to which value set neurons
            rand: `float` radius in which final value can be randomized"""

        for layer in self.LL:
            layer.tune_weight(val, randval)

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


def get_plot(x, y, title=None, name=None):
    """Generates pyplot plot and puts it to the summary

    Uses matplotlib.pyplot to generate plot y(x), saves it to temporary buffer
    and converts it to tensorflow readable format.
    Uses name "writer" for summary writer

    Args:
        x: `iterable`. Function argument, horizontal axis
        y: `iterable`. Function values, vertical axis
        title: `str`. Title of plot
        name: `str`. name scope for tensor ops for convertion

    Returns:
        nuffin

    Raises:
        TypeError if x or y are not iterable or different length"""
    try:
        iter(x)
        iter(y)
    except TypeError:
        raise TypeError('x and y must be iterable')
    if not len(x) == len(y):
        raise TypeError('x and y must be same length')
    default_name = 'plot'
    if not name:
        scope_name = default_name
    else:
        scope_name = name
    plt.figure()
    plt.plot(x, y)
    if not title:
        plt.title = title
    plt.grid()
    imbuf = io.BytesIO()
    plt.savefig(imbuf, format='png')
    imbuf.seek(0)
    with tf.name_scope(scope_name):
        img = tf.image.decode_png(imbuf.getvalue(), channels=4)
        img = tf.expand_dims(img, 0)
        im_op = tf.summary.image(scope_name, img)
    im_sum = sess.run(im_op)
    writer.add_summary(im_sum)
    writer.flush()
    return feeder

def get_feeder(pick, is_training=False):
    """Returns dictionary mapped for placeholders, unique"""
    if is_training:
        return form_feeder(inputs + [tt], [gt_learn, gv_learn, n_learn], pick)
    return form_feeder(inputs + [tt], [gt_test, gv_test, n_test], pick)

def train_net(runs, iterator, training=False):
    """Contains commands for training neural network, unique

    Uses name "writer" for summary writer

    Args:
        runs: `int` how many iterations to perform
        iterator: `int` external cumulative variable for iteration count
        training: `bool` are we training the network or not?

    Returns:
        `int` of iteration count"""
    if not iterator:
        raise RuntimeError('please define iterator. or set it to non-zero')
    for _ in range(runs):
        if training:
            pick = np.random.randint(len(n_learn))
            feeder = get_feeder(pick, training)
            sess.run(train_step, feed_dict=feeder)
            summary = sess.run(mergsumm, feed_dict=feeder)
        else:
            pick = np.random.randint(len(n_test))
            feeder = get_feeder(pick, training)
            summary = sess.run(mergsumm, feed_dict=feeder)
        iterator += 1
        writer.add_summary(summary, iterator)
    writer.flush()
    return iterator

def get_curve():
    """Builds a sequence of ANN output for n_full, unique

    Forms an numpy.array for each value in n_full

    Returns:
        `numpy array` of shape (2, X)"""
    res = np.array([])
    for item in range(len(n_full)):
        feeder = form_feeder(
            inputs + [tt], [gt_full, gv_full, n_full], item)
        res = np.append(res, sess.run(mln_out, feed_dict=feeder))
    return res


data = np.array([])
data_full = np.array([])
data_full_names = np.array([])
with open('data_m.csv') as datafile:
    reader = csv.reader(datafile)
    for row in reader:
        data = np.append(data, row)
data = data.reshape(3, 32).astype(float)  # Три 1-тензора длиной 32
data[0] = data[0]*0.001  # Перевод Вт в кВт
with open('data_ANN_full.csv') as datafile:
    reader = csv.reader(datafile)
    for row in reader:
        data_full = np.append(data_full, row)
data_full = data_full.reshape(23, 20045)
with open('data_ANN_full_names.csv') as datafile:
    reader = csv.reader(datafile)
    for row in reader:
        data_full_names = np.append(data_full_names, row)


# Формируется архив индексов, который потом перемешивается.
# Обучающей, тестовой и валидационной выборке присваиваются одинаковые
# значения. Прирчём соотношения выборок 70:20:10
indx = np.arange(data[0].size)
np.random.shuffle(indx)
n_full = data[0]
gt_full = data[1]
gv_full = data[2]
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
    gt_learn = np.append(gt_learn, data[1][ind])
    gv_learn = np.append(gv_learn, data[2][ind])
for ind in indx[int(np.floor(indx.size*0.7)):int(np.floor(indx.size*0.9))]:
    n_test = np.append(n_test, data[0][ind])
    gt_test = np.append(gt_test, data[1][ind])
    gv_test = np.append(gv_test, data[2][ind])
for ind in indx[int(np.floor(indx.size*0.9)):]:
    n_val = np.append(n_val, data[0][ind])
    gt_val = np.append(gt_val, data[1][ind])
    gv_val = np.append(gv_val, data[2][ind])

inputs = [tf.placeholder(tf.float32, name=item) for item in ['gt', 'gv']]
tt = tf.placeholder(tf.float32, name='target')
mln_layout = [1 for _ in range(2)]
mln_tvn = NMLNetwork(inputs, mln_layout, name='tvn')
mln_out = tf.add_n(mln_tvn.get_out())
# with tf.name_scope('error_sq'):
#     error_sq = tf.squared_difference(tt, mln_out)
#     tf.summary.scalar('error_sq', error_sq)
# with tf.name_scope('error_exp'):
#     error_exp = 1 - tf.exp(-tf.squared_difference(tt, mln_out))
#     tf.summary.scalar('error_exp', error_exp)
# with tf.name_scope('error_abs'):
#     error_abs = (tt - mln_out)/(1 + tf.abs(tt - mln_out))
#     tf.summary.scalar('error_abs', error_abs)
# with tf.name_scope('error_root'):
#     error_root = tf.sqrt(tf.squared_difference(tt, mln_out) - 1) - 1
#     tf.summary.scalar('error_root', error_root)
with tf.name_scope('errors'):
    error_sq = tf.squared_difference(tt, mln_out)
    error_exp = 1 - tf.exp(-tf.squared_difference(tt, mln_out))
    error_abs = (tt - mln_out)/(1 + tf.abs(tt - mln_out))
    error_root = tf.sqrt(tf.squared_difference(tt, mln_out) + 1) - 1
    tf.summary.scalar('error_sq', error_sq)
    tf.summary.scalar('error_exp', error_exp)
    tf.summary.scalar('error_abs', error_abs)
    tf.summary.scalar('error_root', error_root)
tf.summary.scalar('output', mln_out)
with tf.name_scope('weights'):
    tf.summary.scalar('weight00', mln_tvn.LL[0].VN[0].ww[0])
    tf.summary.scalar('weight01', mln_tvn.LL[0].VN[0].ww[1])
    tf.summary.scalar('weight02', mln_tvn.LL[0].VN[0].ww[2])
mergsumm = tf.summary.merge_all()

optim = tf.train.MomentumOptimizer(0.01, 0.9)
# optim = tf.train.GradientDescentOptimizer(0.1)
train_step = optim.minimize(error_root)
sess = tf.Session()
sess.run(tf.global_variables_initializer())
curtime = datetime.strftime(datetime.now(), '%H%M%S')
writer = tf.summary.FileWriter('tbrd_ann/' + curtime, sess.graph)
