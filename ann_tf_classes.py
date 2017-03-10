# -* coding=utf-8 -*-
# Рабочий скрипт нейросети

import numpy as np
import tensorflow as tf


class NNeuron():
    def __init__(self, inputs, type='relu', name=None):
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
        if not(name):
            self.name = self.default_name
        else:
            self.name = name
        self.iws = None
        self.oo = None
        with tf.name_scope(self.name):
            with tf.name_scope('ww') as scope:
                self.ww = [
                    tf.Variable(
                        np.random.rand(),
                        name='ww_{}'.format(item))
                    for item in np.arange(len(inputs))]
            with tf.name_scope('iws') as scope:
                self.iw = [
                        tf.multiply(in_it, ww_it, name='iw')
                        for in_it, ww_it in zip(inputs, self.ww)]
                self.iws = tf.add_n(self.iw, name='sum_of_iw')
            with tf.name_scope('oo') as scope:
                self.oo = tf.nn.relu(self.iws)


class NLayer():
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
        if not(name):
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


class NMLNetwork():
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
        if not(name):
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
