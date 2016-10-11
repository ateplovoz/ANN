# -*- coding: utf-8 -*-
# ANN_classes.py
# Классы для нейросети

import numpy as np


# Функция активации
def actfunc(val):
    if val > 0:
        return val
    else:
        return 0.01 * val


# Производная функции активации
def deractfunc(val):
    return [(0.01, 1)[v > 0] for v in val]


class NSignal:
    def __init__(self, inputs):
        self.v = np.ones([inputs])
        self.g = np.ones([inputs])


class Neuron:
    def __init__(self, cii):
        self.shake = 0.1  # Величина встряхивания весов
        self.pullback = 0.01  # Величина упругого возвращения весов
        self.deltaT = 0.001  # Скорость обучения
        self.expscale = 1  # Крутизна экспоненты
        self.Vii = NSignal(cii)  # Массив входов
        self.Vw = NSignal(cii)  # Массив взвешенных входов
        self.Viws = NSignal(1)  # Суммированный сигнал
        self.Voo = NSignal(1)  # Массив выходных сигналов
        self.Vt = np.array([0])  # Массив целевых значений

    # Функция ошибки
    # 1-e^(-b(t-x)^2)
    def errfunc(self, t, x):
        return 1 - np.exp(-self.expscale * np.power(t - x, 2))

    # Производная функции ошибки
    # -2b(t-x)e^(-b(t-x)^2)
    def dererrfunc(self, t, x):
        return -2 * self.expscale\
                * (t - x)\
                * np.exp(-self.expscale * np.power(t - x, 2))

    # Присваивание градиента
    def getgrad(self, tt=float, appr='direct'):
        self.Vt = tt
        if appr == 'direct':  # Прямой перенос по связям
            self.Voo.g = 1 * self.Vt
        if appr == 'target':  # Вычисление расстояния до цели
            gr = 1 - np.exp(-self.expscale * np.power(self.Voo.v - self.Vt, 2))
            if self.Vt - self.Voo.v > 0:
                self.Voo.g = gr
            else:
                self.Voo.g = -gr

    def forward(self, ii):  # Прямой прогон
        if len(ii) != self.Vii.v.size:
            raise RuntimeError('V-input size mismatch!')
        self.Vii.v = np.array(ii)
        self.Viws.v = np.sum(self.Vii.v * self.Vw.v)
        self.Voo.v = actfunc(self.Viws.v)
        return self.Voo.v

    def backward(self):  # Обратный прогон
        # Получение градиента iws
        self.Viws.g = self.Voo.g.sum()\
                      # * self.deltaT\
                      # * deractfunc([self.Viws.v])[0]\
                      # * self.dererrfunc(self.Vt, self.Viws.v)
        # Получение градиента весов
        self.Vw.g = deractfunc(self.Vii.v)\
            * self.Vii.v\
            * self.Viws.g\
            * self.dererrfunc(self.Vt, self.Vii.v)
            # * self.dererrfunc(self.Vt, self.Vii.v)
        # Получение градиента входов
        self.Vii.g = self.Vw.v * self.Viws.g
        # Корректировка весов
        self.Vw.v += self.Vw.g
        # Упругое возвращение весов к нулю
        self.Vw.v += -self.Vw.g * self.pullback
        # Встряхивание весов
        self.Vw.v += self.Vw.g\
            * self.shake\
            * (np.random.random(len(self.Vw.g)) * 2 - 1)
        return self.Vii.g

    def wgh_tune(self, val, randval):
        self.Vw.v = np.array([val
                              + (np.random.rand() * 2 - 1)
                              * randval for _ in self.Vw.v])


# Получение листа градиентов как их воспринимает нейрон
def unpackgrad(pack):
    unpacked = [sig.g for sig in pack]
    return unpacked


# Получение листа значений как их воспринимает нейрон
def unpackval(pack):
    unpacked = [sig.v for sig in pack]
    return unpacked


class NNetwork:
    def __init__(self, iorder, lorder, oorder):
        # Входной слой нейронов
        self.Lii = [Neuron(1) for _ in xrange(iorder)]
        # Промежуточный слой нейронов
        self.L1 = [Neuron(iorder) for _ in xrange(lorder)]
        # Выходной слой нейронов
        self.Loo = [Neuron(lorder) for _ in xrange(oorder)]
        # Временная переменная хранения входного вектора
        self.Vii = np.array([])
        # Вектор значений выходов входного слоя
        self.VLii = [NSignal(iorder) for _ in xrange(lorder)]
        # Вектор значений выходов промежуточного слоя
        self.VL1 = [NSignal(lorder) for _ in xrange(oorder)]
        # Вектор значений выходов выходного слоя
        self.VLoo = [NSignal(1) for _ in xrange(oorder)]

    # Установка количества переменных во входном слое
    def cfg_input(self, order):
        self.Lii = [Neuron(order) for _ in self.Lii]

    def forward(self, ii):  # Прямой прогон
        self.Vii = np.array(ii)
        for VLii in self.VLii:
            VLii.v = np.array([Neurii.forward(self.Vii)
                               for Neurii in self.Lii])
        for VL1 in self.VL1:
            VL1.v = np.array([NeurL1.forward(v) for NeurL1, v
                              in zip(self.L1, unpackval(self.VLii))])
        for VLoo in self.VLoo:
            VLoo.v = np.array([Neuroo.forward(v) for Neuroo, v
                               in zip(self.Loo, unpackval(self.VL1))])

    def getnetgrad(self, val):  # Запись градиентов в выходные связи
        for neur, val in zip(self.Loo, val):
            neur.getgrad(val, appr='target')

    def backward(self):  # Обратный прогон
        for VL1, neur in zip(self.VL1, self.Loo):
            VL1.g = neur.backward()
        for neur in self.L1:
            [neur.getgrad(sum(g)) for g in unpackgrad(self.VL1)]
        for VLii, neur in zip(self.VLii, self.L1):
            VLii.g = neur.backward()
        for neur in self.Lii:
            [neur.getgrad(sum(g)) for g in unpackgrad(self.VL1)]
        for _, neur in zip(self.Vii, self.Lii):
            neur.backward()

    def nwgh_reset(self, val):
        for neurii, neurl1, neuroo in zip(self.Lii, self.L1, self.Loo):
            neurii.wgh_tune(val, 0)
            neurl1.wgh_tune(val, 0)
            neuroo.wgh_tune(val, 0)

    def nwgh_randomize(self, scale):
        for neurii, neurl1, neuroo in zip(self.Lii, self.L1, self.Loo):
            neurii.wgh_tune(0, scale)
            neurl1.wgh_tune(0, scale)
            neuroo.wgh_tune(0, scale)
