# -* coding=utf-8 -*-
"""Contains tensorflow ANN classes and functions for data learning"""

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
        self.inputsize = len(inputs)
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
                    for item in np.arange(self.inputsize)]
            with tf.name_scope('iws'):
                self.iw = [
                    tf.multiply(in_it, ww_it, name='iw')
                    for in_it, ww_it in zip(inputs, self.ww)]
                self.iws = tf.add_n(self.iw, name='sum_of_iw')
            with tf.name_scope('oo'):
                self.oo = tf.nn.relu(self.iws)

    def tune_weight(self, val, randval, sess):
        """Tunes weights of neuron

        This function sets each weight to given value and after that adds
        random value (pos or neg) within given range.

        Args:
            value: `float` to which value set neurons
            rand: `float` radius in which final value can be randomized
            sess: `session` current session"""
#         with tf.name_scope(self.name):
#             with tf.name_scope('ww'):
#                 self.ww = [
#                     tf.Variable(
#                         val + np.random.rand()*randval,
#                         name='ww_{}'.format(item))
#                     for item in np.arange(self.inputsize)]
        #  probably poorly optimized --- sess.run for each neuron
        sess.run([weight.assign(val + np.random.rand()*randval) for weight in self.ww])


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

    def tune_weight(self, val, randval, sess):
        """Tunes weight for each neuron in layer

        Calls `tune_weight` method from NNeuron objects, passes given
        arguments.

        Args:
            value: `float` to which value set neurons
            rand: `float` radius in which final value can be randomized
            sess: `sess` current session"""
        for neur in self.VN:
            neur.tune_weight(val, randval, sess)


class NMLNetwork():
    """Multi-layer neural network.

    Multi-layer network contains several layers, containing multiple neurons
    each. Only first layer is connected to inputs. Outputs of last layer can
    be summarized to network output."""
    def __init__(self, inputs, tt, layout_list, name=None):
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
            self.LL = [0]*(len(layout_list) + 1)
            lcount = 1
            self.LL[0] = NLayer(inputs, layout_list[0], name='layer0')
            # probably weird behaviour if `layout_list`
            # is short (less than 2 items)
            for num_neurons in layout_list[1:]:
                self.LL[lcount] = NLayer(
                    self.LL[lcount - 1].get_out(),
                    num_neurons, name='layer{}'.format(lcount))
                lcount += 1
            self.LL[-1] = NLayer(self.LL[-2].get_out(), len(tt), name='outerlayer')
            with tf.name_scope('output'):
                self.output = self.LL[-1].get_out()
                for out in self.output:
                    tf.summary.scalar('output', out)
            with tf.name_scope('error'):
                self.error = [tf.sqrt(
                    tf.squared_difference(tt, out) + 1) - 1 for tt, out in zip(tt, self.output)]
                self.errsum = tf.add_n(self.error, name='error_sum')
                for err in self.error:
                    tf.summary.scalar('error', err)
                tf.summary.scalar(self.name + '-errsum', self.errsum)
        with tf.name_scope('LL-out'):
            [tf.summary.scalar('LL-out', item) for item in self.LL[0].get_out()]
            # tf.summary.scalar(self.name + '-errors', self.error)
            # tf.summary.scalar(self.name + '-output', self.output)
        self.mergsum = tf.summary.merge_all()

        self.TDELTA = 0.001
        self.epoch = 0
        self.feeder = {}
        self.sess = None
        self.optim = None
        self.train_step = None

    def get_out(self):
        """returns output of the last layer (deprecated)"""
        # return self.LL[-1].get_out()
        return self.output

    def tune_weight(self, val, randval, sess):
        """Sets new weight value for each neuron within network.

        Calls tune_weight method from NLayer objects. Passes given arguments.
        Args:
            value: `float` to which value set neurons
            rand: `float` radius in which final value can be randomized
            sess: `session` current session"""

        for layer in self.LL:
            layer.tune_weight(val, randval, sess)

    def init_sess(self):
        """initializes and/or resets network variables, optimizer and session.

        Args:
            none

        Returns:
            nuffin"""
        self.epoch = 0
        self.optim = tf.train.MomentumOptimizer(self.TDELTA, 0.9)
        self.train_step = self.optim.minimize(self.errsum)
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        self.writer = tf.summary.FileWriter(
                'tbrd_model_sym/' + datetime.strftime(datetime.now(), '%H%M%S'),
                self.sess.graph)

    def train(self, iters, dsocket):
        """Contains instructions for training neural network

        Args:
            iters: `int` of how many iterations to perform
            dsocket: a `Datasocket` object that forms feeder and contains data

        Raises:
            RuntimeError if self.sess is not initialized
        """
        if not self.sess:
            raise RuntimeError('session is not initialized.'
                               'use init_sess method')
        for _ in range(iters):
            feeder = dsocket.get_feeder()
            self.sess.run(self.train_step, feed_dict=feeder)
            summary = self.sess.run(self.mergsum, feed_dict=feeder)
            self.epoch += 1
            self.writer.add_summary(summary, self.epoch)
        self.writer.flush()

    def calc(self, data, dsocket):
        """calculates the output of NMLNetwork for given data

        uses Datasocket.set_data_multiple method, so data must be appropriate
        tuple pairs as if constructing Datasocket. If confused about names
        refer to Datasocket.get_data() method.

        Args:
            data: `tuple`s of data of form ((name, data), (...), ...) to pass
                as input
            dsocket: `Datasocket` that currently set as input and tt

        Returns:
            list of calculated output values

        Raises:
            RuntimeError if session is not initialized"""
        if not self.sess:
            raise RuntimeError('session is not initialized.'
                               'use init_sess method')
        result = np.array([])
        temp_set = tuple([(
            name,
            dsocket.get_data(name)[0])
            for name in dsocket.get_sock()])
        dsocket.set_data_multiple(*data)
        for indx in range(dsocket.get_len()):
            feeder = dsocket.get_feeder(indx)
            result = np.append(
                    result,
                    self.sess.run(self.output, feed_dict=feeder))
            # summary = self.sess.run(self.mergsum, feed_dict=feeder)
            # self.epoch += 1
            # self.writer.add_summary(summary, self.epoch)
        # self.writer.flush()
        # restore previous data
        dsocket.set_data_multiple(*temp_set)
        result = result.reshape(
                int(result.size/len(self.output)),
                len(self.output))
        return result


class Datasocket():
    """class for managing outer connections of networks

    Provides ways for creating feeders, managing external data and such"""
    def __init__(self, *args, name=None):
        """creates Datasocket object

        Datasocket is a placeholder tensor with functions, that makes data
        management a little bit easier
        TODO: integrate this into NMLNetwork class?

        Args:
            *args: `tuple`s of (name, data_list) that are considered as data
            inputs

        Returns:
            an object of Datasocket class

        Raises:
            TypeError if something in arg is not a tuple"""
        self.sockets = {}
        self.default_name = 'datasocket'
        if not name:
            self.name = self.default_name
        else:
            self.name = name
        with tf.name_scope('datasocket'):
            for item in args:
                if not isinstance(item, tuple):
                    raise TypeError('{} is not a tuple of (name, data)'.format(item))
                socket_name, socket_data = item
                self.sockets[socket_name] = (
                        (tf.placeholder(tf.float32, name=socket_name), socket_data))

    def get_len(self):
        """returns minimum length of data array"""
        dlen = np.inf  # actual infinity, holy sh!t
        for item in self.sockets:
            _, data = self.sockets[item]
            dlen = min(dlen, len(data))
        return dlen

    def get_feeder(self, pick=None):
        """forms dict_feed from all datasockets
        Args:
            pick: `int` index of row. If not provided than random one is
            chosen
        Returns:
            `dict` appropriate to use as dict_feed to feed placeholders
        """
        feeder = {}
        if not pick:
            pick = np.random.randint(self.get_len())
        for item in self.sockets:
            dsock, data = self.sockets[item]
            feeder[dsock] = data[pick]
        return feeder

    def get_sock(self, *args):
        """returns requested tensors

        Args:
            *args: `list of str` of names of tensors that are required.
            If not provided returns all dsockets

        Returns
            `list of Tensors` with requested dsockets"""
        if not args:
            return self.sockets
        res = []
        for item in args:
            try:
                res.append(self.sockets[item][0])
            except KeyError:
                pass
        return res

    def get_data(self, *args):
        """returns data for according tensors

        Args:
            *args: `list of str` of names of tensors which data that is
            required. If not provided returns None

        Returns
            `list of Tensors` with requested dsockets OR None"""
        res = []
        for item in args:
            try:
                res.append(self.sockets[item][1])
            except KeyError:
                pass
        return res

    def set_data(self, name, data):
        """switches feeding data in socket

        placeholder tensor remains the same

        Args:
            name: 'str' name of socket for data replacement
            data: 'iterable' of new data to feed
        Raies:
            KeyError if name doesn't correspond to any names in sockets
        """
        try:
            self.sockets[name] = (self.sockets[name][0], data)
        except KeyError:
            raise KeyError('no such socket {}'.format(name))

    def set_data_multiple(self, *args):
        """sets new data for socket using tuples as in constructor

        tensors remain the same

        Args:
            args: `tuple`s of (name, data) same as if initializing class

        Raises:
            KeyError if any key in args raises KeyError in original
            dictionary"""
        for name, data in args:
            try:
                self.sockets[name]
                self.set_data(name, data)
            except KeyError:
                raise KeyError('original sockdict has no {}'.format(item))


def get_curve(data, inputs, tt):
    """Builds a sequence of ANN output for data
    Forms an numpy.array for each value ordered sequantially as they
    appear in 'data'

    Args:
        data: `list`. Data to map sequence of output. Must be list of
            length of shape [inputs, tt]
        inputs: `list of Tensors` of inputs
        tt: `list of Tensors` of target values

    Returns:
        `numpy array` of network outputs

    Raises:
        TypeError if data is not same shape as inputs + tt or is not list"""
    if not isinstance(data, list):
        raise TypeError('data must be list of shape (inputs + tt)')
    if len(data) != len(inputs + tt):
        raise TypeError('data must be list of shape (inputs + tt)')
    res = np.array([])
    for item in range(len(data)):
        feeder = get_feeder(data, item)
        res = np.append(res, sess.run(mln_out, feed_dict=feeder))
    return res

def get_plot(x, y, writer, title=None, name=None):
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
