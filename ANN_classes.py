# -*- coding: utf-8 -*-
# ANN_classes.py
# Классы для нейросети

import numpy as np


class NSignal:
    def __init__(self, inputs):
        self.v = np.ones([inputs])
        self.g = np.ones([inputs])


class Neuron:
    def __init__(self, cii):
        self.deltaT = 0.01  # Скорость обучения
        self.expscale = 1  # Крутизна экспоненты
        self.Vii = NSignal(cii)  # Массив входов
        self.Vw = NSignal(cii)  # Массив взвешенных входов
        self.Viws = NSignal(1)  # Суммированный сигнал
        self.Voo = NSignal(1)  # Массив выходных сигналов
        self.Vt = np.array([0])  # Массив целевых значений

    def actfunc(self, val):  # Функция активации
        if val > 0:
            return val
        else:
            return 0.01 * val

    def deractfunc(self, val):  # Производная функции активации
        if val > 0:
            return 1
        else:
            return 0.01

    def getgrad(self, tt=float, appr='direct'):  # Присваивание градиента
        self.Vt = tt
        if appr == 'direct':  # Прямой перенос по связям
            self.Voo.g = self.Vt
        if appr == 'target':  # Вычисление расстояния до цели
            gr = 1 - np.exp(-self.expscale*np.power(self.Vt - self.Voo.v, 2))
            if self.Vt - self.Voo.v > 0:
                self.Voo.g = gr
            else:
                self.Voo.g = -gr

    def forward(self, ii):  # Прямой прогон
        if len(ii) != self.Vii.v.size:
            raise RuntimeError('V-input size mismatch!')
        self.Vii.v = np.array(ii)
        self.Viws.v = np.sum(self.Vii.v * self.Vw.v)
        self.Voo.v = self.actfunc(self.Viws.v)
        return self.Voo.v

    def backward(self):  # Обратный прогон
        self.Viws.g = self.deractfunc(self.Viws.v) * self.Voo.g
        self.Vw.g = self.Vii.v * self.Viws.g
        self.Vii.g = self.Vw.v * self.Viws.g
        self.Vw.v += self.Vw.g
        return self.Vii.g


class NNetwork:
    def __init__(self, iorder, lorder, oorder):
        self.Lii = [Neuron(1) for _ in xrange(iorder)]  # Входной слой нейронов
        self.L1 = [Neuron(iorder) for _ in xrange(lorder)]  # Промежуточный слой нейронов
        self.Loo = [Neuron(lorder) for _ in xrange(oorder)]  # Выходной слой нейронов
        self.Vii = np.array([])  # Временная переменная хранения входного вектора
        self.VLii = [NSignal(iorder) for _ in xrange(lorder)]  # Вектор значений выходов входного слоя
        self.VL1 = [NSignal(lorder) for _ in xrange(oorder)]  # Вектор значений выходов промежуточного слоя
        self.VLoo = [NSignal(1) for _ in xrange(oorder)]  # Вектор значений выходов выходного слоя

    def cfg_input(self, order):  # Установка количества переменных во входном слое
        self.Lii = [Neuron(order) for _ in self.Lii]

    def forward(self, ii):  # Прямой прогон
        self.Vii = np.array(ii)
        for VLii in self.VLii:
            VLii.v = np.array([Neurii.forward(self.Vii) for Neurii in self.Lii])
        for VL1 in self.VL1:
            VL1.v = np.array([NeurL1.forward(v) for NeurL1, v in zip(self.L1, self.unpackval(self.VLii))])
        for VLoo in self.VLoo:
            VLoo.v = np.array([Neuroo.forward(v) for Neuroo, v in zip(self.Loo, self.unpackval(self.VL1))])

    def getnetgrad(self, val):  # Запись градиентов в выходные связи
        for neur, val in zip(self.Loo, val):
            neur.getgrad(val, appr='target')

    def unpackgrad(self, pack):  # Получение листа градиентов как их воспринимает нейрон
        unpacked = [sig.g for sig in pack]
        return unpacked

    def unpackval(self, pack):  # Получение листа значений как их воспринимает нейрон
        unpacked = [sig.v for sig in pack]
        return unpacked

    def backward(self):  # Обратный прогон
        for VL1 in self.VL1:
            VL1.g = [neur.backward() for neur in self.Loo]
        for neur in self.L1:
            [neur.getgrad(sum(g)) for g in self.unpackgrad(self.VL1)]
        for VLii in self.VLii:
            VLii.g = [neur.backward() for neur in self.L1]
        for neur in self.Lii:
            [neur.getgrad(sum(g)) for g in self.unpackgrad((self.VL1))]
        for _ in self.Vii:
            [neur.backward() for neur in self.Lii]

